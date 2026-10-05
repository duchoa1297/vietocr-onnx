# vietocr-onnx

Vietnamese OCR models in ONNX format, runnable with `onnxruntime` only (no PyTorch, no internet):

| File | Purpose | Source | License |
|---|---|---|---|
| `det.onnx` | Text line detection | PaddleOCR PP-OCRv6 det small (ONNX build from RapidOCR 3.9.2) | Apache-2.0 |
| `vietocr_encoder.onnx` | Line image (height 32 px) → features | [VietOCR](https://github.com/pbcquoc/vietocr) `vgg_seq2seq`, exported to ONNX | Apache-2.0 |
| `vietocr_decoder.onnx` | One decoding step: previous token + hidden state → next-token log-probabilities | same | Apache-2.0 |
| `vietocr.json` | Vocabulary and preprocessing parameters | same | Apache-2.0 |

Download the files from [Releases](../../releases) (tag `ocr-v1`).

SHA-256:

```
090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f  det.onnx
7375c11ad23d52547a344ec5f53adf89db8302f035ceb3f1e67cff2c04069ae7  vietocr_encoder.onnx
73b92cd370563952b345703ecdce84d8374a9f0a7018afee062170efe04f21f5  vietocr_decoder.onnx
```

## Accuracy

The ONNX export produces exactly the same text as the original PyTorch VietOCR model on 202/202 test lines. On a real
phone scan (CamScanner) of a Vietnamese business notice, detection + recognition read 100% of the accented Vietnamese
words and 98% of the numbers correctly, at about 8 seconds per page on a laptop CPU. An int8-quantised version was tried
and rejected (10x slower on CPU and different output on 12/202 lines).

## Usage (recognition)

```python
import json, numpy as np, onnxruntime as ort
from PIL import Image

meta = json.load(open("vietocr.json", encoding="utf-8"))
enc = ort.InferenceSession("vietocr_encoder.onnx")
dec = ort.InferenceSession("vietocr_decoder.onnx")
i2c = {i + meta["offset"]: c for i, c in enumerate(meta["vocab"])}

def read_line(img: Image.Image) -> str:
    w, h = img.size
    nw = int(np.ceil(int(meta["image_height"] * w / h) / 10) * 10)
    nw = min(max(nw, meta["image_min_width"]), meta["image_max_width"])
    x = np.asarray(img.convert("RGB").resize((nw, meta["image_height"]), Image.LANCZOS), np.float32)
    x = x.transpose(2, 0, 1)[None] / 255.0
    hidden, enc_out = enc.run(None, {"image": x})
    tok, out = np.array([meta["sos"]], np.int64), []
    for _ in range(meta["max_seq_length"]):
        logp, hidden = dec.run(None, {"tgt": tok, "hidden": hidden, "encoder_outputs": enc_out})
        tok = logp.argmax(-1).astype(np.int64)
        if tok[0] == meta["eos"]:
            break
        out.append(int(tok[0]))
    return "".join(i2c.get(t, "") for t in out)
```

Batching works the same way (stack images of equal width). Text detection uses the standard PaddleOCR DB
post-processing (threshold 0.3, box threshold 0.5, unclip ratio 2.0 worked best for Vietnamese diacritics).

## Credits and licenses

- VietOCR, Copyright (c) pbcquoc and contributors, Apache License 2.0: https://github.com/pbcquoc/vietocr
- PaddleOCR, Copyright (c) PaddlePaddle Authors, Apache License 2.0: https://github.com/PaddlePaddle/PaddleOCR
- RapidOCR (ONNX conversion of PaddleOCR models), Apache License 2.0: https://github.com/RapidAI/RapidOCR

The files here are format conversions of those models and are distributed under the same Apache License 2.0 (see
`LICENSE`). Used by the FreeKit personal tool for offline PDF scan → Word conversion.
