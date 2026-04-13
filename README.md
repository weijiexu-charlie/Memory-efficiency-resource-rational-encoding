# Resource-rational-memory-encoding

This is the repository for the ACL 2026 paper "Memory efficiency and resource-rational encoding in sentence processing."


Our models are adapted and trained from nanoGPT ([Karpathy, 2023](https://github.com/karpathy/nanogpt)), and can be downloaded from [this google drive folder](https://drive.google.com/drive/folders/1hjIIgxO5LfufE07D9bzazIOwqyfRRy6d?usp=sharing). Code for Experiment 1 and 2 in the paper is based on these models directly trained from nanoGPT.

We've also uploaded our models to huggingface. Code for Experiment 3 in the paper is based on these huggingface models:

- $\lambda$ = 0.0 ([model](https://huggingface.co/weijiexu97/nanogpt-noisy-lambda0p0-wt103-bs512))
- $\lambda$ = 0.001 ([model](https://huggingface.co/weijiexu97/nanogpt-noisy-lambda0p001-wt103-bs512))
- $\lambda$ = 0.01 ([model](https://huggingface.co/weijiexu97/nanogpt-noisy-lambda0p01-wt103-bs512))
- $\lambda$ = 0.1 ([model](https://huggingface.co/weijiexu97/nanogpt-noisy-lambda0p1-wt103-bs512))
- $\lambda$ = 1.0 ([model](https://huggingface.co/weijiexu97/nanogpt-noisy-lambda1p0-wt103-bs512))
