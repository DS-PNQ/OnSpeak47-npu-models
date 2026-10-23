# OnSpeak47 NPU models

Qualcomm Hexagon NPU (QNN HTP) builds of the on-device models used by
[OnSpeak47](https://github.com/DS-PNQ/OnSpeak47) — offline VI ⇄ EN ⇄ ZH
speech translation. Weights live in [Releases](../../releases), not in git.

All files are **ONNX EPContext models**: precompiled QNN context binaries
wrapped in ONNX, loaded by ONNX Runtime ≥ 1.24 with Qualcomm's QNN plugin
EP. They are SoC-specific.

| SoC | Phone class | HTP | Zipformer ASR | HY-MT 1.5 1.8B |
|---|---|---|---|---|
| `sm8850` | Snapdragon 8 Elite Gen 5 | v81 | yes | yes |
| `sm8750` | Snapdragon 8 Elite | v79 | yes | — |
| `sm8650` | Snapdragon 8 Gen 3 | v75 | yes | — |

Built with QAIRT **2.51** (`onnxruntime-qnn` 2.7.0). Load them with
`com.qualcomm.qti:onnxruntime-android-qnn:2.7.0` +
`com.qualcomm.qti:qnn-runtime:2.51.0` (or newer runtime).

## Files

**Zipformer 2e streaming transducer** (vi/en/zh, FP16 on HTP, 100 % of nodes
on the NPU): `zipformer_{encoder,decoder,joiner}.<soc>.ctx.onnx`.
Encoder input `x [1,39,80]` Kaldi fbank + 35 cache tensors (batch fixed to 1).
Use the `tokens.txt` of the source model.

**HY-MT 1.5 1.8B** (INT8 block-128 weights kept from the source export,
FP16 activations, prefill + decode graphs sharing one weight copy):
`hymt_ar64.sm8850.ctx.onnx` + `hymt_ar1.sm8850.ctx.onnx` (embedding lookup on
CPU, everything else one NPU partition each) and the shared context binary
`hymt_ar64.sm8850.ctx_qnn.bin` (1.88 GB) they both reference. Keep all three
in one folder; load both sessions with `ep.share_ep_contexts=1`.
**Use v1.0.3-npu or later** for HY-MT. Earlier HY-MT binaries decode garbage
on device: v1.0.0 kept the float32-min attention-mask fill (not representable
in FP16); v1.0.1 fixed that, but QNN's RmsNorm on the K heads (HunYuan's
`key_layernorm`) produced head 0 only, so every attention score was wrong;
v1.0.2 fixed the norms, but the HTP's `ArgMax` still flipped bit 16 of token
ids above 65535 (EOS read as " goats"). v1.0.3 takes the argmax over two
60409-wide halves. Zipformer files are unchanged since v1.0.0. v1.0.4 (app graph REV 5) compiles the
decode graph over a 256-token cache instead of 1024, so every decode step moves
less data; prompts that do not fit decode with the prefill graph.

Graph I/O (cache length 1024, AR = 64 or 1): `input_ids [1,AR]`,
`attention_mask [1,1024]`, `position_ids [1,AR]`,
`past_key_values.{0..31}.{key,value}` fp16 `[1,4,1024-AR,128]` →
`next_ids [1,AR]` (greedy argmax), `new.{i}.{key,value}` fp16 `[1,4,AR,128]`.
The caller owns the cache; see `backend/translation_hymt_npu.py` in the app
repo for the reference loop.

## Hy-MT2 (v2.0.0-npu)

Release **v2.0.0-npu** carries the same three HY-MT files built from Tencent
**Hy-MT2-1.8B** (Apache-2.0, same architecture and I/O as HY-MT 1.5; app graph
**REV 6**, used by the `com.omnivoice.onspeak47.hymt2` app). The Zipformer
files are byte-identical to v1.0.4. Do not mix HY-MT files from different
releases. License: `LICENSE-HY-MT2.txt`.

**v2.0.1-npu** (app graph **REV 7**) runs Hy-MT2 entirely on the NPU: the
embedding lookup is a quantized HTP Gather, so the `.ctx.onnx` wrappers are a
single EPContext node each (no CPU table copy), and it adds
`hymt_ar64_256.sm8850.ctx.onnx`, a 64-token prefill over a 256-slot cache for
prompts that fit (cache I/O `[1,4,192,128]`, `attention_mask [1,256]`). Four
HY-MT files: `hymt_ar64`, `hymt_ar64_256`, `hymt_ar1` wrappers + the shared
`hymt_ar64.sm8850.ctx_qnn.bin` (2.14 GB).

## Hy-MT2 1.25-bit (v2.1.0-npu)

**v2.1.0-npu** (app graph **REV 8**, used by the `com.omnivoice.onspeak47.hymt125`
app) carries the same four HY-MT files and I/O as v2.0.1, built from Tencent
**Hy-MT2-1.8B-1.25Bit-GGUF** (AngelSlim ternary QAT, Apache-2.0). Its linear
weights are ternary with one fp16 scale per 256 inputs, which is exactly a
2-bit `MatMulNBits`; the HTP keeps them at 2 bits, so the shared
`hymt_ar64.sm8850.ctx_qnn.bin` is 0.94 GB (v2.0.1: 2.14 GB). The tied
embedding / LM head (Q6_K in the GGUF) is 8-bit. The conversion is exact
(ONNX greedy output = the GGUF weights in fp32); translation quality is that
of the 1.25-bit model, below the 8-bit Hy-MT2 of v2.0.1. Zipformer files are
byte-identical to v2.0.1. License: `LICENSE-HY-MT2-1.25Bit.txt`.

## Qwen3-ASR code-switching ASR (v2.2.0-npu)

**v2.2.0-npu** adds a second ASR (app graph **REV 1** of
`QwenAsrNpuEngine`, `com.omnivoice.onspeak47.hymt125` app): **Qwen3-ASR-0.6B**
with a vi/en/zh code-switching LoRA (merged at 0.5), vocabulary pruned to
87,193 tokens, 8-bit weights, FP16 activations, sm8850 only. Five files, keep
them in one folder: `qasr_enc.sm8850.ctx.onnx` (audio encoder, one 8 s block
per run, 235 MB), `qasr_ar64.sm8850.ctx.onnx` + `qasr_ar1.sm8850.ctx.onnx`
(decoder prefill / decode wrappers, load both with `ep.share_ep_contexts=1`)
over `qasr_ar64.sm8850.ctx_qnn.bin` (570 MB), and `qasr_assets.bin` (CPU side:
int8 embedding table, vocabulary bytes, mel filters; 91 MB).

Graph I/O: encoder `mel [1,128,800]` + `mask [1,1,1,104]` -> `feats [1,104,1024]`;
decoder `x [1,AR,1024]` (embeddings, audio rows in place), `pos [1,AR]`,
`mask [AR,512+AR]`, `past_{k,v}_{0..27}` fp16 `[1,8,512,128]` -> `next_ids [1,AR]`
(pruned-vocab argmax), `new_k`/`new_v` fp16 `[28,8,AR,128]`. Reference loop:
`backend/streaming_asr/qwen3asr_npu.py` in the app repo. Files are independent
of the HY-MT / Zipformer files of v2.1.0-npu (not repeated here).

## Reproduce

In the app repo: `optimize/16_qnn_compile.py zipformer --soc all` and
`optimize/16_qnn_compile.py hymt`; Qwen3-ASR: `optimize/18_qwen3asr.py export` +
`optimize/16_qnn_compile.py qwen3asr`. Research notes and verification status:
`docs/qualcomm_npu.md`.

## Licenses

- HY-MT files: Tencent HY Community License — see `LICENSE-HY-MT.txt` and
  `NOTICE`. **Not licensed in the EU, UK or South Korea.**
- Zipformer files: CC BY-NC-SA 4.0 (non-commercial, share-alike), from
  [lmcu000's zh-en-vi stage-2e model](https://huggingface.co/lmcu000/zipformer-streaming-zh-en-vi-2e).
- Qwen3-ASR files: Apache-2.0 — see `LICENSE-Qwen3-ASR.txt` and `NOTICE`.
