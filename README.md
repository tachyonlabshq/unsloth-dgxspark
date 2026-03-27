# Unsloth DGX Spark — Optimized Docker Image

Custom Unsloth image for NVIDIA DGX Spark (Grace Blackwell GB10, ARM64).

**Base chain:** `nvcr.io/nvidia/pytorch` (NGC) -> `unsloth/unsloth:dgxspark-latest` -> this image

## What this adds over `unsloth/unsloth:dgxspark-latest`

- **Unsloth Studio web UI** — no-code training, Data Recipes, Model Arena (port 8000)
- **Latest transformers** — Qwen3.5, Llama 3.x, and other new architectures
- **flash-linear-attention** — 12x faster MoE routing (critical for Qwen3.5)
- **causal-conv1d** — fast path for SSM/Mamba-style layers
- **Latest unsloth + trl + peft** — newest training optimizations

## Services (managed by supervisord)

| Service | Port | Description |
|---------|------|-------------|
| **Unsloth Studio** | 8000 | No-code training web UI |
| **Jupyter Lab** | 8888 | Notebook-based training |
| **SSH** | 22 | Remote shell access |

## Usage

```bash
docker pull ghcr.io/tachyonlabshq/unsloth-dgxspark:latest
```

Or in docker-compose.yml:
```yaml
image: ghcr.io/tachyonlabshq/unsloth-dgxspark:latest
```

## Auto-updates

- **GitHub Actions** rebuilds every Monday at 04:00 UTC
- **Watchtower** on the DGX Spark pulls new images nightly at 03:00

Packages are updated weekly; the server picks up changes within 24 hours.

## Optimal Training Settings (Qwen3.5-9B on DGX Spark)

```
batch_size:   2
grad_accum:   1
seq_len:      2048
lora_r:       64
lora_alpha:   128
dtype:        BF16
packing:      true
optim:        adamw_8bit
FLA:          required (included in this image)
```

Achieves ~0.38 steps/sec (~1,550 tok/s) with zero swap on 128GB unified memory.

## Build locally

```bash
docker build -t unsloth-dgxspark .
```

## DGX Spark Notes

- Requires `privileged: true` in Docker for GPU access
- Set `TORCH_CUDA_ARCH_LIST=12.1` for Blackwell SM 12.1
- Set `HF_HUB_DISABLE_XET=1` (xet transfer fails on ARM64)
- 128 GB unified memory shared between CPU and GPU
