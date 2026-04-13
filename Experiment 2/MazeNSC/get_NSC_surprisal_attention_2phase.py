'''
Before running this script, make sure you have:
1. Downloaded the NSC corpus and the Maze RT data from https://github.com/vboyce/natural-stories-maze
2. Saved the text file ("natural_stories_sentences.tsv") under the folder "nsc_corpus"

This script will output both the surprisal and the attention-score files in the same "nsc_corpus" folder
'''

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

NSC_TEXTFILE = "nsc_corpus_data/natural_stories_sentences.tsv"
OUTPUT_FILE = "nsc_corpus_data/NSC_lm_features.csv"
ATTN_DIR = "nsc_corpus_data/NSC_attn" 

SEED = 1568
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

N_SAMPLES = 100

model_paths = {
    'base_gpt2': '../trained_models/ckpt_16000_gpt2_base.pt',
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

def process_NSC_df(df):
    '''
    Preprocess NSC text data, explode the df s.t. each row correspond to a word.
    '''
    df['word'] = df['Sentence'].str.split()
    df = df.explode('word').reset_index(drop=True)
    df = df.rename(columns={"Story_Num": "story_id",
                            "Sentence_Num": "sentence_id",
                            "Sentence": "sentence"})
    df['word_id'] = df.groupby(['story_id','sentence_id']).cumcount()
    df = df[["story_id", "sentence_id", "word_id", "word", "sentence"]]
    df.insert(0, "global_word_id", df.index.values)

    return df


def tokenize_df(df):
    '''
    Tokenize each word with a preword whitespace.
    '''
    df["hftoken"] = df.word.apply(lambda x: tokenizer.tokenize(" " + x))
    df = df.explode("hftoken", ignore_index=True)
    df["token_id"] = df.hftoken.apply(tokenizer.convert_tokens_to_ids)
    df = df.reset_index(drop=True)
    df["attn_idx"] = df.index.astype(np.int64)
    return df


def prepare_tokens_data(df, context_len):
    '''
    Do this only within one coherent story.
    '''
    # tokenized_df = tokenize_df(df)
    token_ids = df.token_id.tolist()
    pad_id = eot # end of text token in gpt2
    if tokenizer.pad_token_id is not None:
        pad_id = tokenizer.pad_token_id

    data = torch.full((len(token_ids), context_len + 1), pad_id, dtype=torch.long)
    for i in range(len(token_ids)):
        example_tokens = token_ids[max(0, i - context_len) : i + 1]
        data[i, -len(example_tokens) :] = torch.tensor(example_tokens)
    print(f"Data has a shape of: {data.shape}")	

    return data

######################## Get surprisal and attention for corpus ########################

def pack_batch_attention_fixed(attn_maps, batch_ids, out_array, token_indices, context_len, nan_rows_mask=None):
    """
    Save attention rows at qpos = T-2 into a [L, H, N_tokens, context_len] memmap.
    """
    import numpy as np
    import torch

    B, T = batch_ids.shape
    qpos = T - 2

    # per-row real keys count (after trimming)
    valid_len = (batch_ids != pad_id).sum(dim=1)
    K_each = valid_len - 1
    Kmax = int(min(int(K_each.max().item()), T, context_len))
    if Kmax <= 0:
        return

    # normalize token_indices
    token_indices = token_indices.detach().cpu().numpy() if isinstance(token_indices, torch.Tensor) else np.asarray(token_indices)
    H = attn_maps[0].shape[1]

    # rows to force NaN
    if nan_rows_mask is None:
        nan_rows_mask = np.zeros((B,), dtype=bool)
    else:
        nan_rows_mask = np.asarray(nan_rows_mask, dtype=bool)

    for l, amap in enumerate(attn_maps):
        amap_q = amap[:, :, qpos, :Kmax].contiguous().float()     # (B, H, Kmax)
        amap_q = torch.nan_to_num(amap_q, nan=0.0, posinf=0.0, neginf=0.0)
        amap_q = amap_q.detach().cpu().numpy().astype(np.float32)

        packed = np.full((B, H, context_len), np.nan, dtype=np.float16)

        for b in range(B):
            if nan_rows_mask[b]:
                continue

            K = int(min(int(K_each[b].item()), Kmax))
            if K <= 0:
                continue

            vec = amap_q[b, :, :K]                             
            # renormalize per head across K
            denom = vec.sum(axis=-1, keepdims=True)
            np.maximum(denom, 1e-12, out=denom)
            vec = (vec / denom).astype(np.float16)

            prev_len = K - 1
            if prev_len > 0:
                start = context_len - 1 - prev_len
                end = context_len - 1
                packed[b, :, start:end] = vec[:, :prev_len]
            packed[b, :, context_len - 1] = vec[:, prev_len] 

        out_array[l, :, token_indices, :] = packed


def run_story_write_attn(df_story, model, model_name, attn_mm):
    """
    Two-phase, so no padding:
      For the first 512 tokens, compute surprisal and attention for every token using exact prefix lengths
      For tokens beyond 512, use a fixed 512-length window (last 512 tokens)
    """
    tokens = df_story["token_id"].tolist()
    attn_idx = df_story["attn_idx"].to_numpy()  # global row indices
    N = len(tokens)
    W = context_len + 1        

    story_trueprob = np.full((N,), np.nan, dtype=np.float64)
    story_entropy = np.full((N,), np.nan, dtype=np.float64)

    def _masked_mean(stack_np): 
        mask = ~np.isnan(stack_np)
        cnt = mask.sum(axis=0)
        ssum = np.where(mask, stack_np, 0.0).sum(axis=0)
        return np.divide(ssum, cnt, out=np.full(cnt.shape, np.nan), where=cnt > 0)

    @find_executable_batch_size(starting_batch_size=32)
    def inference_loop(batch_size):
        accelerator.free_memory()
        print(f"{model_name} | story {int(df_story.story_id.iloc[0])} | batch_size={batch_size}")

        def run_fixed_length_batch(batch_ids, batch_attn_idx):
            
            B, L = batch_ids.shape
            is_first_token = (L == 1)

            # Base Models:
            if model_name in ['base_gpt2']:
                logits, _, attn_maps = model(batch_ids, return_attn=True)

                if not is_first_token:
                    probs = torch.softmax(logits[:, -2, :], dim=-1)         # predict last token
                    true_ids = batch_ids[:, -1]
                    tp = probs[torch.arange(B, device=probs.device), true_ids]
                    ent = torch.distributions.Categorical(probs=probs).entropy()

                    pos = np.searchsorted(attn_idx, batch_attn_idx)
                    story_trueprob[pos] = tp.detach().cpu().numpy()
                    story_entropy[pos] = ent.detach().cpu().numpy()

                pack_batch_attention_fixed(
                    attn_maps, batch_ids, attn_mm, batch_attn_idx, context_len,
                    nan_rows_mask=(np.ones(B, dtype=bool) if is_first_token else None)
                )
                return

            # Noisy Models (average over samples):
            probs_samples = []
            ent_samples = []
            attn_accum = None

            for _ in range(N_SAMPLES):

                logits, _, attn_maps = model(batch_ids, return_attn=True)

                if not is_first_token:
                    probs = torch.softmax(logits[:, -2, :], dim=-1)
                    true_ids = batch_ids[:, -1]
                    tp = probs[torch.arange(B, device=probs.device), true_ids]
                    ent = torch.distributions.Categorical(probs=probs).entropy()
                    probs_samples.append(tp.detach().cpu().numpy())
                    ent_samples.append(ent.detach().cpu().numpy())

                if attn_accum is None:
                    attn_accum = [am.detach().float().clone() for am in attn_maps]
                else:
                    for i in range(len(attn_accum)):
                        attn_accum[i] += attn_maps[i].detach().float()

            # averaged surprisal/entropy
            if not is_first_token and len(probs_samples) > 0:
                mean_tp = _masked_mean(np.stack(probs_samples, axis=0))   
                mean_en = _masked_mean(np.stack(ent_samples,   axis=0))  
                pos = np.searchsorted(attn_idx, batch_attn_idx)
                story_trueprob[pos] = mean_tp
                story_entropy[pos] = mean_en

            # average attention maps across samples
            for i in range(len(attn_accum)):
                attn_accum[i] /= float(N_SAMPLES)

            pack_batch_attention_fixed(
                attn_accum, batch_ids, attn_mm, batch_attn_idx, context_len,
                nan_rows_mask=(np.ones(B, dtype=bool) if is_first_token else None)
            )

        ########## Phase 1: exact prefixes for first 512 tokens ##########
        Lmax = min(W, N) 
        buckets, bucket_idxs = {}, {}
        for i in range(Lmax):
            L = i + 1
            buckets.setdefault(L, []).append(tokens[:L])
            bucket_idxs.setdefault(L, []).append(attn_idx[i])

        for L in range(1, Lmax + 1):
            seqs = buckets[L]; idxs = bucket_idxs[L]
            if not seqs: continue
            for s in range(0, len(seqs), batch_size):
                e = min(s + batch_size, len(seqs))
                sub = torch.tensor(seqs[s:e], dtype=torch.long, device=device) 
                sub_idx = np.asarray(idxs[s:e])
                run_fixed_length_batch(sub, sub_idx)

        ########## Phase 2: sliding window beyond 512 tokens ##########
        if N > W:
            seqs, idxs = [], []
            for i in range(W, N):
                seqs.append(tokens[i - W + 1 : i + 1])
                idxs.append(attn_idx[i])

            for s in range(0, len(seqs), batch_size):
                e = min(s + batch_size, len(seqs))
                sub = torch.tensor(seqs[s:e], dtype=torch.long, device=device) 
                sub_idx = np.asarray(idxs[s:e])
                run_fixed_length_batch(sub, sub_idx)

        return story_trueprob, story_entropy

    probs, ents = inference_loop()

    out_df = df_story.copy()
    out_df[f"entropy_{model_name}"] = ents
    out_df[f"surp_{model_name}"] = -np.log2(probs)

    return out_df


##########################################################################

def run_all():
    os.makedirs(ATTN_DIR, exist_ok=True)

    NSC_df = pd.read_csv(NSC_TEXTFILE, sep='\t')
    NSC_processed_df = process_NSC_df(NSC_df)
    tokenized_df = tokenize_df(NSC_processed_df)   
    base_df = tokenized_df.copy()          

    story_order = pd.unique(tokenized_df["story_id"])
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

        # Process stories and write attention directly into the memmap
        dfs_out = []
        for sid in story_order:
            story_df = tokenized_df[tokenized_df["story_id"] == sid]
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
            "axis_names": ["layer", "head", "token(attn_idx)", "vector(≤512 right-aligned)"]
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