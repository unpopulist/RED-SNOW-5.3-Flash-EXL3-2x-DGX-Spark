# Credits

This recipe is a thin layer of instructions. It only works because of the work below. The recipe's own text and
scripts are licensed under the Apache License 2.0 ([`LICENSE`](LICENSE)). Everything listed here keeps its own
license. No model weights are part of this repository: they are downloaded from their sources.

## Models

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

## Engine and kernels

- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart and the TensorFold contributors is the
  inference engine the fork is based on. It is under the Apache License 2.0 from v0.6.0; code written before then
  keeps its MIT notice.
- **[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)**
  by MiaAI-Lab and its contributors (Apache License 2.0) is the patch series for two-Spark GLM-5.3-Flash serving
  that the fork ports. It covers multi-stream decoding, the FP8 cache, copy drafts, prefill and decode kernels, the
  launcher design, the container image recipe and much more. The fork's `THIRD_PARTY_NOTICES.md` names the
  contributors of each ported patch: abhicnv007, Alexbob0, ezoushen, gabewillen, desy0305, johnwhited, meleesciony,
  m-naoki-m, ThomasWadeZ, BadAd84, wojo, jdecker76 and others credited there.
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
  (Apache License 2.0) published RED-SNOW measurements on the same hardware, which the README cites for comparison.

## Runtime

The engine's image is based on NVIDIA's PyTorch container (`nvcr.io/nvidia/pytorch:26.07-py3`). The NVIDIA software
in it is governed by NVIDIA's license terms, which the container prints at start.

If something is missing or credited wrongly, please open an issue.
