# RED-SNOW-5.3-FLASH on two DGX Sparks

> **Point your agents at this file: [`AGENTS.md`](AGENTS.md).** It is a step-by-step setup, from two bare Sparks
> to a validated server, written for an AI coding agent (and readable by people).

This recipe serves [RED-SNOW-5.3-FLASH EXL3 SAGE 4.91 bpw](https://huggingface.co/vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw)
on two NVIDIA DGX Sparks. The weights are split with tensor parallelism (TP=2), and the two GPUs exchange data over
the Sparks' QSFP link (RoCE). The engine is a fork of [TensorFold](https://github.com/ashhart/TensorFold) that carries
the port of this mixed-width EXL3 pack. The [DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) drafter
supplies speculative drafts. Rank 0 serves an OpenAI-compatible API on port 8000.

This repository has only the instructions and a thin wrapper (`run.sh`). The engine, its image build and its
launcher live in the engine fork, under `deploy/redsnow`. Its `deploy/redsnow/README.md` is the full reference for
every setting.

**Model use.** RED-SNOW's [model card](https://huggingface.co/Blackfrost-AI/RED-SNOW-5.3-FLASH-BF16) asks that it be
used only by experienced operators, in environments they own or are explicitly authorized to assess, with a human
in control of scope, targets, credentials, tool execution and destructive actions. Read the card before you deploy.

## Requirements

**Hardware**

- Two DGX Sparks (GB10, 128 GB unified memory), joined by a QSFP cable. Each end needs an IPv4 address on the
  RoCE netdev (on DGX OS, typically `enp1s0f0np0`, RDMA device `rocep1s0f0`).
- Nothing else running on either GPU. Each rank holds ~92 GiB of weights, and the launcher wants ~115 GiB
  `MemAvailable` at start (it refuses below 110 GiB).
- Disk: ~200 GB per node, as follows.
  - Head: the 200 GB pack, plus ~100 GB while it writes rank 1's half.
  - Worker: ~100 GB for rank 1's half.
  - Both nodes: ~45 GB for the image and 2.2 GB for the drafter.

**Software**

- DGX OS (Ubuntu 24.04), an NVIDIA driver with CUDA 13 support, and Docker with the NVIDIA Container Toolkit and
  `buildx`, on both nodes. `/dev/infiniband/uverbs0` must be present.
- Passwordless ssh from the head to the worker, with the same `$HOME` path on both nodes.
- `git`, `rsync`, the `hf` CLI (`pip install -U huggingface_hub`) and optionally `zstd`, on the head.

## Quick start

```bash
# 1. the engine fork, at a release tag (on the head)
git clone --branch redsnow-v1.0.0 https://github.com/unpopulist/TensorSnow <engine-dir>

# 2. this recipe
git clone https://github.com/unpopulist/RED-SNOW-5.3-Flash-EXL3-2x-DGX-Spark && cd RED-SNOW-5.3-Flash-EXL3-2x-DGX-Spark
cp .env.example .env      # set ENGINE_DIR, HEAD_ADDR, WORKER_ADDR, WORKER_SSH

# 3. the image, on both nodes
./run.sh build

# 4. model + drafter on both nodes (AGENTS.md, step 5), then:
./run.sh check            # every launch check, starts nothing
tmux new -s redsnow ./run.sh start
./run.sh stop
```

Smoke test, once `start` prints `serving`:

```bash
curl -s http://<rank0-host>:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "RED-SNOW-5.3-FLASH", "max_tokens": 64, "messages": [{"role": "user", "content": "Reply with OK."}]}'
```

## Configuration highlights

These are the defaults. Change any of them in `.env`.

| Setting | Default | |
|---|---|---|
| `CONTEXT` | 327680 | Prompt + reply window per request (320K tokens). |
| `PARALLEL`, `POOL_STREAMS` | 4, 1 | Four concurrent requests decoded in shared rounds. They share one window's cache pool. |
| `KV_DTYPE` | fp8 | FP8 latent cache. Use `bf16` for exact results with a shorter window. |
| `MEMORY_RESERVE_GIB` | 14.5 | Memory left free at start, as headroom for peaks under load. |
| `DRAFTER` | DFlash2 | Plus prompt-lookup ("copy") drafts. Drafts change speed, not replies. `none`: no DFlash2. |
| `COMM` | roce | Small all-gathers as one-shot RDMA writes (b12x's RoCE transport), large ones over NCCL. |
| `SERVED_NAME`, `PORT` | `RED-SNOW-5.3-FLASH`, 8000 | |

## Measured performance

Measured on two DGX Sparks (GB10) with the defaults: RED-SNOW 4.91 bpw, FP8 KV, DFlash2 + copy drafts, 4 streams
sharing one cache pool, a 320K-token window per request and a 14.5 GiB memory reserve.

| | |
|---|---|
| Decode, one request (thinking off) | prose ~37 tok/s, code ~60, JSON ~72, edit/quote ~95 |
| Decode, four requests at once | ~68 tok/s aggregate |
| Prefill | ~1,100–1,150 tok/s: 55K tokens in 46–49 s, 236K in 214 s, 306K in 282 s (needle found) |
| Lowest free memory under load | ~6–9 GiB per node |
| Startup | ~2.5 minutes to a ready API |

### Benchmark: v1.0.0 vs the next release

Both images were measured with the same defaults (320K window, 4 streams), one request at a time. The method is the
benchmark suite from [justincasey/RED-SNOW-5.3-Flash-2x-DGX-Spark](https://github.com/justincasey/RED-SNOW-5.3-Flash-2x-DGX-Spark)
(adapted there from [vcruz305's GLM-5.3 recipe](https://github.com/vcruz305/GLM-5.3-EXL3-DGX-Spark-recipe), MIT),
run unchanged.

- **Decode:** six prompts with thinking on, greedy, 512 tokens. Speed is the median of 3 drafted runs, counting
  the server's decode time only.
- **Long prompt:** a 127,712-token Project Gutenberg book (Moby-Dick).
- **Checks:** every drafted reply matched the undrafted reply token for token, and all checks passed (25/25).

- **v1.0.0:** tag `redsnow-v1.0.0`, the current release.
- **Next release:** adds MiaAI-Lab patches 0090, 0099–0107 and `FILL_ROWS=2048`. It is not published yet.

| Decode (tok/s) | v1.0.0 | Next release |
|---|---|---|
| code | 46.2 | 46.6 |
| prose | 35.9 | 36.3 |
| math | 55.8 | 56.3 |
| chat_explain | 38.2 | 38.7 |
| chat_multiturn | 44.9 | 45.5 |
| long_doc (3K-token prompt) | 41.7 | 42.9 |
| **Mean, drafted** | **43.8** | **44.4** |
| Mean, no drafting | 26.8 | 26.8 |

| Long prompt (127,712 tokens) | v1.0.0 | Next release |
|---|---|---|
| Time to first token | 115.4 s | 113.6 s |
| Prefill | 1,107 tok/s | 1,124 tok/s |
| Decode after it, drafted / no drafting | 39.5 / 26.3 tok/s | 40.2 / 26.1 tok/s |

The next release gains little in this one-request benchmark. Most of its new patches speed up several streams
decoding at long context, which this benchmark does not measure.

## Known limits

- **Tight startup memory margin at 320K.** The window fits with only a few tenths of a GiB to spare. The launcher
  waits for free memory to settle before it starts. If the engine still refuses the window, wait a minute and retry,
  or lower `CONTEXT`.
- **Four streams share one pool.** A long request leaves less room for the others. When the pool is full, the
  youngest stream is requeued and replayed.
- **DFlash2 is non-commercial** (CC BY-NC-ND 4.0). For commercial use, serve with `DRAFTER=none` (slower) or obtain a
  license from Inco AI.
- The spill tier (kept prompts on disk) and the display-memory KV reservation are not enabled.

## Next steps

A 4-bit latent KV cache is planned. It would store symmetric int4 values with one fp16 scale per group of 32 (288
bytes per latent row, against 528 for FP8), and keep the indexer keys in FP8. The goal is a ~580K-token shared window
on two Sparks (against ~350K with FP8), for 256–512K-token contexts. The format was chosen by measuring real
RED-SNOW cache rows: attention-output error was ~0.75%, against ~0.26% for FP8 and ~1.1% for a per-row int4 scale.
The kernels are not built yet. Quality will be validated with needle and long-document tests before it becomes an
option.

## Credits and licenses

This recipe is a thin layer of instructions. It only works because of the work below. The recipe's own text and
scripts are licensed under the Apache License 2.0 ([`LICENSE`](LICENSE)). Everything listed here keeps its own
license. No model weights are part of this repository: they are downloaded from their sources.

### Models

- **[GLM-5.3-Flash](https://huggingface.co/zai-org)** by Z.AI (zai-org) is the base model, under the MIT License.
- **[RED-SNOW-5.3-FLASH](https://huggingface.co/Blackfrost-AI/RED-SNOW-5.3-FLASH-BF16)** by
  [Blackfrost-AI](https://huggingface.co/Blackfrost-AI) is the finetune this recipe serves, under the MIT License.
  **Usage notice** (from its model card, paraphrased): the model is meant for experienced operators, working in
  environments they own or are explicitly authorized to assess. A human operator stays in control of scope, target
  selection, credentials, tool execution and destructive actions. Treat its tool calls as proposals. Read
  [the card](https://huggingface.co/Blackfrost-AI/RED-SNOW-5.3-FLASH-BF16) before use.
- **[RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw](https://huggingface.co/vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw)**
  by [vcruz305](https://huggingface.co/vcruz305), in collaboration with Blackfrost-AI, is the mixed-width EXL3
  pack served here, under the MIT License.
- **[GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)** by
  [Inco AI](https://huggingface.co/incoai) is the speculative drafter, under
  **[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)**: non-commercial use only, no derivatives.
  Commercial licensing is available from Inco AI (see its model card). `DRAFTER=none` serves without it. Inco AI
  asks that you cite:

  ```bibtex
  @misc{inco2026dflash2,
    title  = {{DFlash 2: Keep Drafting Parallel}},
    author = {{Inco AI}},
    year   = {2026},
    month  = {August},
    url    = {https://inco.ai/blog/dflash2/}
  }

  @inproceedings{chen2026dflash,
    title     = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
    author    = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
    booktitle = {International Conference on Machine Learning (ICML)},
    year      = {2026}
  }
  ```

### Engine and kernels

- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart and the TensorFold contributors is the
  inference engine the fork is based on. It is under the Apache License 2.0 from v0.6.0; code written before then
  keeps its MIT notice.
- **[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)**
  by MiaAI-Lab and its contributors (Apache License 2.0) is the patch series for two-Spark GLM-5.3-Flash serving
  that the fork ports. It covers multi-stream decoding, the FP8 cache, copy drafts, prefill and decode kernels, the
  launcher design, the container image recipe and much more. The fork's `THIRD_PARTY_NOTICES.md` names the
  contributors of each ported patch: abhicnv007, Alexbob0, ezoushen, gabewillen, desy0305, johnwhited, meleesciony,
  m-naoki-m, ThomasWadeZ, BadAd84, wojo, jdecker76, mikolaj92 and others credited there.
  - Being ported for the next release:
    - **[BadAd84](https://github.com/BadAd84)** wrote patches 0099 (tokenizing without the GIL), 0100 (draft
      prelaunch), 0101 (one-pass prompt selection), 0103 (decode matmuls without clusters), 0105 (prompt indexer
      scoring in CUDA) and 0107 (decode indexer), plus the `FILL_ROWS=2048` default.
    - **Lukasz Raczylo ([lukaszraczylo](https://github.com/lukaszraczylo))** wrote patch 0090, the expert decode
      launch order, in [PR #95](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/95).
- **[b12x](https://github.com/local-inference-lab/b12x)** by local-inference-lab provides the "RoCEnante" one-shot
  RoCE all-gather transport (`COMM=roce`), under the Apache License 2.0. Its C proxy is used unchanged, and its
  GPU kernel is reimplemented in CUDA C++.
- **[jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark)** by Jay Leaton
  (Apache License 2.0) is the source of two adaptations, both by way of MiaAI-Lab's patches: the L2 prefetch
  kernel (adapted from its patch 0460) and the 16-byte expert weight loads (an idea from its patch 0580). Parts of
  the tool-call handling come from its patch 0620.
- **[ExLlamaV3](https://github.com/turboderp-org/exllamav3)** by turboderp (MIT License) defines the EXL3 format the
  pack is stored in, and provided the reference the port's layers were checked against.
- **[justincasey/RED-SNOW-5.3-Flash-2x-DGX-Spark](https://github.com/justincasey/RED-SNOW-5.3-Flash-2x-DGX-Spark)**
  (Apache License 2.0), another RED-SNOW recipe for two DGX Sparks: we compared this recipe against its published
  measurements while building it, and its benchmark suite (`bench/`, adapted from vcruz305's, MIT) measured the
  numbers in [Benchmark](#benchmark-v100-vs-the-next-release). No code or settings from it are part of this recipe.

### Runtime

The engine's image is based on NVIDIA's PyTorch container (`nvcr.io/nvidia/pytorch:26.07-py3`). The NVIDIA software
in it is governed by NVIDIA's license terms, which the container prints at start.

If something is missing or credited wrongly, please open an issue.
