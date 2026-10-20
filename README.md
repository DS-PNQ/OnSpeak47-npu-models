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

## Reproduce

In the app repo: `optimize/16_qnn_compile.py zipformer --soc all` and
`optimize/16_qnn_compile.py hymt`. Research notes and verification status:
`docs/qualcomm_npu.md`.

## Licenses

- HY-MT files: Tencent HY Community License — see `LICENSE-HY-MT.txt` and
  `NOTICE`. **Not licensed in the EU, UK or South Korea.**
- Zipformer files: CC BY-NC-SA 4.0 (non-commercial, share-alike), from
  [lmcu000's zh-en-vi stage-2e model](https://huggingface.co/lmcu000/zipformer-streaming-zh-en-vi-2e).
