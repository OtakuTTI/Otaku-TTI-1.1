# Otaku TTI 1.1

<img width="520" height="520" alt="strong_warrior" src="https://github.com/user-attachments/assets/ace771d4-4fdc-4d2c-b3f3-41e0cee85a0e" />


**A lightweight native 512×512 anime text-to-image latent diffusion engine.**

Otaku TTI 1.1 brings native 512×512 anime generation to consumer hardware, thin laptops, and edge devices. Trained end-to-end under a strict 60W power budget on an Acer Nitro V 15 (NVIDIA GeForce RTX 4050 Laptop GPU, 6GB VRAM), Otaku TTI proves that expressive, high-resolution generative AI can be developed sustainably without mega-datacenter clusters.

---

## Highlights

- **Native 512×512 Latent Diffusion**: 4× spatial resolution increase over Otaku TTI 1.0, generating rich details, vivid anime eyes, layered clothing, and expansive landscapes.
- **Ultra-Low Memory Footprint**: Runs comfortably in under **1.4 GB GPU VRAM** (FP16 weights are only **332 MB**), making it accessible for budget 2GB–6GB laptops, older GPUs, and everyday CPUs.
- **Blazing Fast**: Generates a 512×512 artwork in **~1.14 to 1.45 seconds** on an entry-level laptop GPU using the DPM-Solver++ multistep scheduler (25 steps).
- **Built-in Japanese Tag Translation**: Native support for Japanese anime tags (e.g., `初音ミク`, `笑顔`, `青空`, `草原`, `最高品質`) via an offline tag mapping dictionary (`tag_map_ja.json`).
- **Clean Standalone Engine**: 100% self-contained inference code, stripped of training dependencies and optimizer states.

---

## Specifications

| Parameter | Specification |
| :--- | :--- |
| **Model Series** | Otaku TTI 1.1 |
| **Architecture** | Latent Diffusion Model (LDM) |
| **UNet Parameters** | ~166 Million |
| **Latent Space** | 4 channels, 64×64 (Native 512×512 RGB via 8× VAE) |
| **Text Encoder** | OpenAI CLIP ViT-Base/32 (Cross-Attention Dim: 512) |
| **VAE** | StabilityAI SD-VAE-ft-mse |
| **Sampling Algorithm** | DPM-Solver++ (2nd order multistep, 20–25 steps) |
| **VRAM Consumption** | ~1.4 GB peak during 512×512 generation |
| **Release Weights** | `weights/otaku_tti_1.1_fp16.pt` (332.8 MB) <br> `weights/otaku_tti_1.1.pt` (665.3 MB) |

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/your-username/otaku_tti_1.1.git
cd otaku_tti_1.1
pip install -r requirements.txt
```

### 2. Generating Images

#### English / Booru Tag Prompt
```bash
python3 generate.py -p "1girl, solo, hatsune miku, teal twintails, smiling, anime screencap, masterpiece, 512x512" -o miku.png
```

#### Japanese Prompt (Auto-Translated)
```bash
python3 generate.py -p "初音ミク, 笑顔, 青空, 草原, 最高品質" -o miku_japanese.png
```

#### Pure Landscape / Scenery (Zero Characters)
```bash
python3 generate.py \
  -p "no humans, scenery, landscape, meadow, green grass, blue sky, clouds, studio ghibli aesthetic" \
  -np "human, 1girl, 1boy, person, signature, text, watermark" \
  -o landscape.png
```

#### Batch Generation
```bash
python3 generate.py -p "1girl, solo, anime aesthetic, fantasy armor, masterpiece" -n 4 -o grid.png
```

---

## CLI Options

| Argument | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--prompt` | `-p` | *(Miku default)* | Text prompt (Booru tags, natural language, or Japanese) |
| `--negative-prompt` | `-np` | `"low quality, ..."` | Negative prompt for Classifier-Free Guidance |
| `--checkpoint` | `-c` | *(Auto-detects FP16)* | Path to weights file (`.pt`) |
| `--output` | `-o` | `output.png` | Output path for saved image or grid |
| `--num-images` | `-n` | `1` | Number of images to generate |
| `--steps` | `-s` | `25` | DPM-Solver++ denoising steps |
| `--cfg` | | `7.5` | Classifier-Free Guidance scale |
| `--seed` | | `None` | Random seed for reproducibility |
| `--device` | | `cuda` (if available) | Compute device (`cuda` or `cpu`) |

---


## License

This project, its code, and model weights are released under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC-BY-NC 4.0)**.

- **Non-Commercial**: Free for personal, research, artistic, and educational use. Commercial use or monetization is prohibited without prior written permission.
- **Attribution**: You must give appropriate credit when redistributing or modifying this work.
