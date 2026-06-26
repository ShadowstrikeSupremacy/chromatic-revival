# Chromatic Revival
### AI-Powered Image Processing Suite
 
A local desktop application combining deep-learning colorization, neural-network OCR, and image format conversion — all running fully offline with no API keys or cloud dependencies.

---

## Features

### Colorize
Convert black-and-white photographs to color using the Zhang et al. Caffe model (Lab color space, trained on ImageNet). Includes SSIM, PSNR, and colorfulness quality metrics.

### Extract Text (OCR)
Two-stage document OCR powered by **docTR** (DBNet detection + CRNN recognition). Accuracy comparable to Google Vision / AWS Textract on printed and scanned documents. Output is plain selectable text reconstructed in reading order. Export as JSON.

### Convert Format
Lossless and lossy conversion between **JPEG, PNG, BMP, TIFF, WEBP** with per-file size comparison. JPEG/WEBP quality is adjustable (1–100).

---

## Quick Start

```bash
git clone https://github.com/ShadowstrikeSupremacy/chromatic-revival
cd chromatic-revival
pip install -r requirements.txt
python run.py gui
```

Model files required in `models/`:
- `colorization_release_v2.caffemodel`
- `colorization_deploy_v2.prototxt`
- `pts_in_hull.npy`

OCR model weights (~165 MB total) download automatically to `~/.cache/doctr/models/` on first use.

---

## CLI Usage

```bash
# Colorize a single image
python run.py colorize photo.jpg

# Batch colorize a folder
python run.py batch ./old_photos --output ./colorized

# Extract text (prints to stdout)
python run.py ocr document.png

# Extract text and save structured JSON
python run.py ocr document.png -o results.json

# Convert image format
python run.py convert photo.png --to jpeg --quality 90
python run.py convert photo.png --to webp -o compressed.webp
```

---

## How It Works

### Colorization — Zhang et al. (2016)
```
Grayscale input
    → Lab color space (L channel only)
    → Caffe CNN predicts a + b channels (224×224)
    → Upsample ab to original resolution
    → Merge L + ab → RGB output
```
The model was trained to predict plausible colors from luminance alone. Quality is measured with SSIM (structural similarity) and PSNR.

### OCR — docTR two-stage pipeline
```
Image input
    → CLAHE contrast enhancement (preserves color for recognition)
    → DBNet (db_resnet50): detects word bounding boxes
    → CRNN (crnn_vgg16_bn): reads characters in each box
    → Words sorted into reading order (Block → Line → Word)
    → Plain text output
```
All inference runs locally via PyTorch on CPU (or GPU if available).

### Format Conversion — PIL
Handles alpha-channel stripping for JPEG, LZW compression for TIFF, and quality control for JPEG/WEBP. Reports original vs converted file size.

---

## Project Structure

```
chromatic-revival/
├── src/
│   ├── colorizer.py      # Caffe colorization engine
│   ├── gui.py            # Tkinter GUI (Colorize / OCR / Convert modes)
│   ├── ocr.py            # StandaloneOcrPipeline (docTR)
│   └── utils.py          # Image utilities and metrics
├── models/
│   ├── colorization_deploy_v2.prototxt
│   ├── colorization_release_v2.caffemodel
│   └── pts_in_hull.npy
├── output/               # Saved colorized images
├── run.py                # CLI entry point
└── requirements.txt
```

---

## Requirements

```
opencv-python>=4.8.1.78
numpy>=1.26.0
matplotlib>=3.8.0
Pillow>=10.2.0
scikit-image>=0.22.0
tqdm>=4.66.1
python-doctr[torch]>=0.9.0
```

Python 3.10+ recommended. PyTorch (CPU build) is installed automatically as a docTR dependency.

---

## Tech Stack

| Component | Library | Purpose |
|---|---|---|
| Colorization | OpenCV DNN + Caffe | Lab-space ab channel prediction |
| OCR detection | docTR / DBNet (ResNet-50) | Word localisation |
| OCR recognition | docTR / CRNN (VGG-16) | Character sequence reading |
| Image I/O | Pillow | Format conversion, display |
| GUI | Tkinter + ttk | Cross-platform desktop UI |
| Metrics | scikit-image | SSIM, PSNR evaluation |

---

## Hosting

This is a Tkinter desktop application — it cannot be hosted as-is on Render or similar platforms. To deploy it as a web app, the GUI layer would need to be rewritten in **Gradio** or **Streamlit** (the backend `colorizer.py` and `ocr.py` modules work unchanged). HuggingFace Spaces offers free hosting for Gradio apps.

---

## References

- Zhang et al., *Colorful Image Colorization* (ECCV 2016)
- Liao et al., *Real-time Scene Text Detection with Differentiable Binarization* (AAAI 2020)
- Shi et al., *An End-to-End Trainable Neural Network for Image-based Sequence Recognition* (TPAMI 2017)
- docTR by Mindee — https://github.com/mindee/doctr
