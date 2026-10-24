# RetiScan Neural Network Architecture

Architecture-only code for the Stage A measurement engine (vessel segmentation
+ artery/vein classification). No training loop, dataset loading, or
optimizer code is included here by design — this is just the model.

## Files

```
retiscan_model/
├── models/
│   ├── __init__.py          exposes RetiScanUNet, GradCAM, etc.
│   ├── resnet_encoder.py    ResNet34 encoder, returns 5 skip-connection feature maps
│   ├── unet_decoder.py      U-Net decoder that upsamples back to full resolution
│   ├── retiscan_unet.py     Combines encoder + decoder + segmentation head
│   └── gradcam.py           Grad-CAM explainability hook
├── check_architecture.py    Sanity check: builds the model, runs one dummy
│                             forward pass, checks output shapes. Not training.
└── README.md
```

## How this maps to the two-stage training plan

```python
from models import RetiScanUNet

# Stage 0 — auxiliary pretraining on Massachusetts Roads (road vs. background)
model = RetiScanUNet(num_classes=2, pretrained_encoder=True)
# ... train, then torch.save(model.state_dict(), "stage0_weights.pt")

# Stage 1 — fine-tune on DRIVE/HRF/AV-DRIVE (background / artery / vein)
model = RetiScanUNet(num_classes=3, pretrained_encoder=True)
model.load_pretrained_stage0("stage0_weights.pt")
# ... fine-tune from here
```

`load_pretrained_stage0()` only copies weights whose name **and** shape
match, so it's safe to call even though Stage 0 and Stage 1 use a different
`num_classes` — the mismatched final layer (`segmentation_head`) is simply
skipped and keeps its fresh random initialization.

## Output shape

`model(x)` returns raw logits of shape `(batch, num_classes, H, W)`.
Softmax / argmax is intentionally left out of `forward()` so the same
logits can feed a loss function (e.g. `CrossEntropyLoss`) directly during
training, and your own post-processing (AVR calculation, visualization)
during inference.

## Checking it works

```bash
pip install torch torchvision
python check_architecture.py
```

This builds the model and runs one dummy forward pass to confirm the
encoder/decoder channel wiring and Grad-CAM hook are correct. It does not
download real data or train anything.

> Note: this code was written and reviewed carefully, but wasn't executed
> in a live PyTorch environment before delivery (no GPU/internet in this
> session). Run `check_architecture.py` first before building on top of it.
