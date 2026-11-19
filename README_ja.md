# Otaku TTI 1.1 (日本語ドキュメント)

**軽量・ネイティブ512×512 アニメ特化テキスト画像生成（Latent Diffusion）エンジン**

Otaku TTI 1.1 は、日常的なノートパソコン、省電力GPU、CPU環境でも高速に動作するネイティブ512×512のアニメ画像生成モデルです。Acer Nitro V 15（NVIDIA GeForce RTX 4050 Laptop GPU, 6GB VRAM）の約60Wという限られた省電力環境下で丹念にトレーニングされ、巨大なデータセンターを使用することなく持続可能（Green AI）に開発されました。

---

## 主な特徴

- **ネイティブ 512×512 解像度**: Otaku TTI 1.0比で4倍の空間解像度。生き生きとした瞳、繊細な髪の流れ、緻密な背景を描画。
- **超軽量 VRAM 消費量**: 生成時のピークVRAMはわずか **約1.4 GB**。軽量FP16モデルは **332 MB** のため、エントリー層のゲーミングノートPCや旧世代GPUでも快適に動作します。
- **高速生成**: DPM-Solver++ マルチステップスケジューラ（25ステップ）を採用し、ラップトップGPUでも **約1.1〜1.4秒** で1枚の高解像度アートを生成。
- **日本語タグ自動翻訳内蔵**: `tag_map_ja.json` を搭載しており、`初音ミク`、`笑顔`、`青空`、`草原`、`最高品質` などの日本語タグをそのままプロンプトとして入力可能。
- **完全に独立したコード**: トレーニング用の余分な依存関係やオプティマイザ状態を一切排除したクリーンな推論専用パッケージ。

---

## スペック概要

| 項目 | 仕様 |
| :--- | :--- |
| **モデル名** | Otaku TTI 1.1 |
| **アーキテクチャ** | 潜在拡散モデル（Latent Diffusion Model） |
| **UNet パラメータ数** | 約1億6600万（166M） |
| **潜在空間サイズ** | 4チャンネル, 64×64（8倍VAEによりネイティブ512×512 RGBにデコード） |
| **テキストエンコーダ** | OpenAI CLIP ViT-Base/32（Cross-Attention次元: 512） |
| **VAE** | StabilityAI SD-VAE-ft-mse |
| **サンプラー** | DPM-Solver++（2次マルチステップ, 20〜25ステップ） |
| **ピークVRAM** | 約1.4 GB（512×512生成時） |
| **公開モデル重み** | `weights/otaku_tti_1.1_fp16.pt` (332.8 MB) <br> `weights/otaku_tti_1.1.pt` (665.3 MB) |

---

## クイックスタート

### 1. 依存ライブラリのインストール

```bash
git clone https://github.com/your-username/otaku_tti_1.1.git
cd otaku_tti_1.1
pip install -r requirements.txt
```

### 2. 画像の生成

#### 日本語タグによる生成（自動英訳対応）
```bash
python3 generate.py -p "初音ミク, 笑顔, 青空, 草原, 最高品質" -o miku_japanese.png
```

#### 英語 / Booruタグによる生成
```bash
python3 generate.py -p "1girl, solo, hatsune miku, teal twintails, smiling, anime screencap, masterpiece, 512x512" -o miku.png
```

#### 人物なしの純粋な風景・背景生成
```bash
python3 generate.py \
  -p "no humans, scenery, landscape, meadow, green grass, blue sky, clouds, studio ghibli aesthetic" \
  -np "human, 1girl, 1boy, person, signature, text, watermark" \
  -o landscape.png
```

#### 複数枚の一括生成
```bash
python3 generate.py -p "1girl, solo, anime aesthetic, fantasy armor, masterpiece" -n 4 -o grid.png
```

---

## 主なオプション引数

| 引数 | 短縮 | デフォルト値 | 説明 |
| :--- | :--- | :--- | :--- |
| `--prompt` | `-p` | 初音ミク | 生成プロンプト（日本語、英語、Danbooruタグ対応） |
| `--negative-prompt` | `-np` | `"low quality, ..."` | ネガティブプロンプト（品質低下や不要要素の抑制） |
| `--checkpoint` | `-c` | *(自動検出)* | モデル重みファイル（`.pt`）のパス |
| `--output` | `-o` | `output.png` | 保存先ファイル名 |
| `--num-images` | `-n` | `1` | 生成する画像枚数 |
| `--steps` | `-s` | `25` | サンプリングステップ数 |
| `--cfg` | | `7.5` | CFGガイダンススケール |
| `--seed` | | `None` | 乱数シード値（再現用） |
| `--device` | | `cuda` (利用可能時) | 計算デバイス (`cuda` または `cpu`) |

---

## ライセンス

本プロジェクト、コード、およびモデル重みは **Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC 4.0)** のもとで公開されています。

- **非商用利用限定**: 個人利用、研究、アート活動、教育目的において自由に使用・改変が可能です。商用利用は事前の許可が必要です。
- **クレジット表示**: 配布または二次利用の際は、著作権表示およびライセンスへのリンクを保持してください。
