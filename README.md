# Otaku TTI 1.0

A lightweight text-to-image anime generation engine.

## Features
- **Compact Architecture**: 166M parameter UNet paired with CLIP and Pretrained VAE.
- **Low Memory**: Operates under 1.2 GB of GPU VRAM or on standard CPU.
- **High Speed**: Generates images in ~200ms on modern GPUs.

## Quickstart

### Installation
```bash
pip install -r requirements.txt
```

### Generating Images
```bash
python3 generate.py -p "1girl, solo, anime aesthetic" -n 4
```

## License
Attribution-NonCommercial 4.0 International License.
