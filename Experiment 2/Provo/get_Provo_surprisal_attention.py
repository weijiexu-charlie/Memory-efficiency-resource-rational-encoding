
import os
import torch
import pandas as pd
import numpy as np
import random
import json
from numpy.lib.format import open_memmap
from transformers import AutoTokenizer
import tiktoken
from modeling_GPT2Noisy import NoisyGPT, NoisyGPTConfig
from modeling_GPT2 import GPT, GPTConfig
from torch.nn import functional as F
from accelerate import Accelerator, find_executable_batch_size

PROVO_TEXTFILE = "provo_corpus_data/Provo_corpus_text.csv"
OUTPUT_FILE = "provo_corpus_data/Provo_lm_features.csv"
ATTN_DIR = "provo_corpus_data/Provo_attn" 

SEED = 1568
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

N_SAMPLES = 100

model_paths = {
    'base_gpt2': '../trained_models/ckpt_16000_gpt2.pt',
    'lambda0p0': '../trained_models/ckpt_16000_lambda0.0.pt',
    'lambda0p001': '../trained_models/ckpt_16000_lambda0.001.pt',
    'lambda0p01': '../trained_models/ckpt_16000_lambda0.01.pt',
    'lambda0p1': '../trained_models/ckpt_16000_lambda0.1.pt',
    'lambda1p0': '../trained_models/ckpt_16000_lambda1.0.pt'
}
model_names = ['base_gpt2', 'lambda0p0', 'lambda0p001', 'lambda0p01', 'lambda0p1', 'lambda1p0']

context_len = 511 # training block size 512 - 1

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")

accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Use pretrained gpt2 tokenizer for all models
enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token
pad_id = eot

######################## Load Models ########################

def load_model(ckpt_path, arch, device=device):
    ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    if arch in ['base_gpt2']:
        cfg = GPTConfig(**ckpt["model_args"])
        model = GPT(cfg)
    else:
        cfg = NoisyGPTConfig(**ckpt["model_args"])
        model = NoisyGPT(cfg)

    state_dict = ckpt["model"]
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

######################## Process corpus ########################

def tokenize_df(df):
    '''
    Tokenize each word with a preword whitespace.
    '''
    df["hftoken"] = df.Word.apply(lambda x: tokenizer.tokenize(" " + x))
    df = df.explode("hftoken", ignore_index=True)
    df["token_id"] = df.hftoken.apply(tokenizer.convert_tokens_to_ids)
    df = df.reset_index(drop=True)
    df["attn_idx"] = df.index.astype(np.int64)
    return df


def prepare_tokens_data(df, context_len):
    '''
    Do this only within one coherent story.
    '''
    token_ids = df.token_id.tolist()
    pad_id = eot # end of text token in gpt2
    if tokenizer.pad_token_id is not None:
        pad_id = tokenizer.pad_token_id

    data = torch.full((len(token_ids), context_len + 1), pad_id, dtype=torch.long)
    for i in range(len(token_ids)):
        example_tokens = token_ids[max(0, i - context_len) : i + 1]
        # data[i, -len(example_tokens) :] = torch.tensor(example_tokens) # padding to the left
        data[i, :len(example_tokens)] = torch.tensor(example_tokens) # padding to the right
    # print(f"Data has a shape of: {data.shape}")	

    return data

######################## Get surprisal and attention for corpus ########################

def pack_batch_attention_fixed(attn_maps, batch_ids, out_array, token_indices, context_len):
    
    batch_ids_cpu = batch_ids.detach().to("cpu", non_blocking=True) if batch_ids.is_cuda else batch_ids
    B, T = batch_ids_cpu.shape

    valid_len = (batch_ids_cpu != pad_id).sum(dim=1)    
    qpos_each = (valid_len - 2).clamp_min(-1)           

    Kfixed = int(context_len)
    vec_cap = out_array.shape[-1]

    for l, amap in enumerate(attn_maps):
        attn_l = amap  # (B, H, T, S), already softmaxed
        H = attn_l.size(1)
        S = attn_l.size(3)

        packed = np.full((B, H, Kfixed), np.nan, dtype=np.float16)

        for b in range(B):
            out_idx = int(token_indices[b])
            qpos = int(qpos_each[b].item())
            if qpos < 0:
                continue

            K = min(qpos + 1, S, Kfixed)
            if K <= 0:
                continue

            vec = (
                attn_l[b, :, qpos, :K]
                .detach()
                .to("cpu")
                .float()
                .numpy()
                .astype("float16")
            )

            prev_len = K - 1 
            if prev_len > 0:
                start = Kfixed - 1 - prev_len
                end = Kfixed - 1
                packed[b, :, start:end] = vec[:, :prev_len]
            packed[b, :, Kfixed - 1] = vec[:, prev_len]

        out_array[l, :, token_indices, :Kfixed] = packed
        if vec_cap > Kfixed:
            out_array[l, :, token_indices, Kfixed:vec_cap] = np.nan


def run_story_write_attn(df_story, model, model_name, attn_mm):

    data = prepare_tokens_data(df_story, context_len)  
    story_rowidx = df_story["attn_idx"].to_numpy()    

    def _select_last_and_prev_positions(batch_cpu):
        valid_len = (batch_cpu != pad_id).sum(dim=1)
        tgt_idx = valid_len - 1
        prev_idx = valid_len - 2
        keep_mask = valid_len >= 2
        return valid_len, tgt_idx, prev_idx, keep_mask

    def _gather_rowwise_logits(logits, row_pos_idx):
        B = logits.size(0)
        ar = torch.arange(B, device=logits.device)
        return logits[ar, row_pos_idx, :]

    @find_executable_batch_size(starting_batch_size=64)
    def inference_loop(batch_size):
        accelerator.free_memory()
        if int(df_story.Text_ID.iloc[0]) % 5 == 0:
            print(f"{model_name} | story {int(df_story.Text_ID.iloc[0])} | batch_size={batch_size}")

        dl = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

        all_probs = []
        all_ents = []

        cursor = 0

        with torch.inference_mode():
            for batch in dl:
                batch_cpu = batch
                valid_len, tgt_idx, prev_idx, keep_mask = _select_last_and_prev_positions(batch_cpu)
                B = batch_cpu.size(0)

                probs = np.full(B, np.nan, dtype=np.float32)
                ents = np.full(B, np.nan, dtype=np.float32)

                if not torch.any(keep_mask):
                    all_probs.append(probs)
                    all_ents.append(ents)
                    cursor += B
                    continue

                batch_sel = batch_cpu[keep_mask]
                valid_len_sel = valid_len[keep_mask]
                tgt_idx_sel = tgt_idx[keep_mask]
                prev_idx_sel = prev_idx[keep_mask]
                global_idx_sel = torch.from_numpy(story_rowidx[cursor:cursor + B])[keep_mask]

                T_sub = int(valid_len_sel.max().item())
                batch_sel = batch_sel[:, :T_sub].to(device, non_blocking=True)
                prev_idx_dev = prev_idx_sel.to(device)
                tgt_idx_dev = tgt_idx_sel.to(device)
                true_ids = batch_sel[torch.arange(batch_sel.size(0), device=device), tgt_idx_dev]

                if model_name in ['base_gpt2']:
                    logits, loss, attn_maps = model(batch_sel, return_attn=True)
                    logits_prev = _gather_rowwise_logits(logits, prev_idx_dev)
                    probs_prev = torch.softmax(logits_prev, dim=-1)
                    tp = probs_prev[torch.arange(probs_prev.size(0)), true_ids]
                    ent = torch.distributions.Categorical(probs=probs_prev).entropy()

                    probs[keep_mask.cpu().numpy()] = tp.detach().cpu().numpy()
                    ents[keep_mask.cpu().numpy()] = ent.detach().cpu().numpy()

                    pack_batch_attention_fixed(
                        attn_maps, batch_sel, attn_mm, global_idx_sel, context_len
                    )

                else:
                    probs_samples, ent_samples = [], []
                    attn_accum = None

                    for _ in range(N_SAMPLES):
                        logits, loss, attn_maps = model(batch_sel, return_attn=True)
                        logits_prev = _gather_rowwise_logits(logits, prev_idx_dev)
                        probs_prev = torch.softmax(logits_prev, dim=-1)
                        tp = probs_prev[torch.arange(probs_prev.size(0)), true_ids]
                        ent = torch.distributions.Categorical(probs=probs_prev).entropy()

                        probs_samples.append(tp.detach().cpu().numpy())
                        ent_samples.append(ent.detach().cpu().numpy())

                        if attn_accum is None:
                            attn_accum = [am.detach().float().clone() for am in attn_maps]
                        else:
                            for i in range(len(attn_accum)):
                                attn_accum[i] += attn_maps[i].detach().float()

                    for i in range(len(attn_accum)):
                        attn_accum[i] /= float(N_SAMPLES)

                    probs[keep_mask.cpu().numpy()] = np.mean(probs_samples, axis=0)
                    ents[keep_mask.cpu().numpy()] = np.mean(ent_samples, axis=0)

                    pack_batch_attention_fixed(
                        attn_accum, batch_sel, attn_mm, global_idx_sel, context_len
                    )

                all_probs.append(probs)
                all_ents.append(ents)
                cursor += B

        probs = np.concatenate(all_probs)
        ents = np.concatenate(all_ents)
        return probs, ents

    probs, ents = inference_loop()

    out_df = df_story.copy()
    out_df[f"entropy_{model_name}"] = ents
    out_df[f"surp_{model_name}"] = np.where(np.isnan(probs), np.nan, -np.log2(probs))
    return out_df

##########################################################################

def run_all():
    os.makedirs(ATTN_DIR, exist_ok=True)

    Provo_df = pd.read_csv(PROVO_TEXTFILE)
    tokenized_df = tokenize_df(Provo_df)   
    base_df = tokenized_df.copy()  

    story_order = pd.unique(tokenized_df["Text_ID"])
    N_tokens = len(tokenized_df)

    for model_name in model_names:
        print(f"\n=== START {model_name} ===")
        model = load_model(model_paths[model_name], arch=model_name, device=device)

        cfg = getattr(model, "config", None)
        n_layers = int(getattr(cfg, "n_layer", len(getattr(model.transformer, "h", []))))
        n_heads = int(getattr(cfg, "n_head", getattr(model.transformer.h[0].attn, "n_head", 0)))

        attn_path = os.path.join(ATTN_DIR, f"attn_{model_name}.npy")
        attn_mm = open_memmap(attn_path, mode="w+", dtype=np.float16,
                              shape=(n_layers, n_heads, N_tokens, context_len))
        attn_mm[:] = 0.0

        dfs_out = []
        for sid in story_order:
            story_df = tokenized_df[tokenized_df['Text_ID'] == sid]
            if len(story_df) == 0:
                continue
            out_df = run_story_write_attn(story_df, model, model_name, attn_mm)
            dfs_out.append(out_df)

        meta = {
            "model": model_name,
            "shape": [n_layers, n_heads, int(N_tokens), int(context_len)],
            "L": int(n_layers),
            "H": int(n_heads),
            "N_tokens": int(N_tokens),
            "context_len": int(context_len),
            "axis_names": ["layer", "head", "token(attn_idx)", "vector(≤511 right-aligned, self at -1)"]
        }
        with open(attn_path + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[{model_name}] wrote attention array: {attn_path}  with meta: {attn_path+'.meta.json'}")

        model_df = pd.concat(dfs_out, axis=0).set_index("attn_idx").sort_index()
        cols_to_add = [c for c in model_df.columns if c.startswith("entropy_") or c.startswith("surp_")]
        base_df = base_df.join(model_df[cols_to_add], on="attn_idx")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    base_df.to_csv(OUTPUT_FILE, index=False)
    print("\nWrote CSV:", OUTPUT_FILE)


if __name__ == "__main__":
    run_all()