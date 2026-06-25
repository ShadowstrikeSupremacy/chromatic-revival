"""
Gradio web interface for Chromatic Revival.

Three tabs:
  1. Colorize     — B&W → colour via Zhang et al. Caffe model
  2. Extract Text — local OCR via docTR (DBNet + CRNN)
  3. Convert      — reformat images between JPEG / PNG / BMP / TIFF / WEBP
"""

import json
import os
import tempfile

import cv2 as cv
import gradio as gr
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Lazy-loaded singletons — models are large; only initialise on first use
# ---------------------------------------------------------------------------

_colorizer = None
_ocr_pipeline = None


def _get_colorizer():
    global _colorizer
    if _colorizer is None:
        from src.colorizer import ImageColorizer
        _colorizer = ImageColorizer()
    return _colorizer


def _get_ocr():
    global _ocr_pipeline
    if _ocr_pipeline is None:
        from src.ocr import StandaloneOcrPipeline
        _ocr_pipeline = StandaloneOcrPipeline(use_gpu=False)
    return _ocr_pipeline


# ---------------------------------------------------------------------------
# Helper: group OCR words into reading-order lines
# ---------------------------------------------------------------------------

def _group_into_lines(results: list) -> list[str]:
    if not results:
        return []
    avg_h = sum(r["bbox"][3] - r["bbox"][1] for r in results) / len(results)
    threshold = max(avg_h * 0.6, 4)
    sorted_words = sorted(results, key=lambda r: (r["bbox"][1] + r["bbox"][3]) / 2)
    lines: list = [[sorted_words[0]]]
    for word in sorted_words[1:]:
        y_center = (word["bbox"][1] + word["bbox"][3]) / 2
        last_y = (lines[-1][-1]["bbox"][1] + lines[-1][-1]["bbox"][3]) / 2
        if abs(y_center - last_y) <= threshold:
            lines[-1].append(word)
        else:
            lines.append([word])
    for line in lines:
        line.sort(key=lambda r: r["bbox"][0])
    return [" ".join(w["text"] for w in line) for line in lines]


# ---------------------------------------------------------------------------
# Tab 1 — Colorize
# ---------------------------------------------------------------------------

def colorize(image: np.ndarray):
    if image is None:
        return None, "Upload an image to colorize."
    try:
        colorizer = _get_colorizer()
    except FileNotFoundError as exc:
        return None, f"Model not found: {exc}\nSee models/README.md for download instructions."

    bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    colorized_bgr = colorizer.colorize_image(bgr)
    colorized_rgb = cv.cvtColor(colorized_bgr, cv.COLOR_BGR2RGB)

    m = colorizer.evaluate_colorization(bgr, colorized_bgr)
    info = (
        f"SSIM: {m['ssim']:.4f}  |  "
        f"PSNR: {m['psnr']:.2f} dB  |  "
        f"Colorfulness: {m['colorfulness']:.2f}"
    )
    return colorized_rgb, info


# ---------------------------------------------------------------------------
# Tab 2 — OCR
# ---------------------------------------------------------------------------

def extract_text(image: np.ndarray):
    if image is None:
        return None, "Upload an image to extract text from.", ""

    pipeline = _get_ocr()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        Image.fromarray(image).save(tmp_path)
        enhanced = pipeline.optimize_image_for_ocr(tmp_path)
        results = pipeline.extract_text(enhanced)
    finally:
        os.unlink(tmp_path)

    if not results:
        return enhanced, "No text detected.", ""

    plain_text = "\n".join(_group_into_lines(results))
    json_str = json.dumps(results, indent=2, ensure_ascii=False)
    return enhanced, plain_text, json_str


# ---------------------------------------------------------------------------
# Tab 3 — Format conversion
# ---------------------------------------------------------------------------

def convert_format(image: np.ndarray, fmt: str, quality: int):
    if image is None:
        return None, "Upload an image to convert.", None

    ext_map = {"JPEG": ".jpg", "PNG": ".png", "BMP": ".bmp", "TIFF": ".tif", "WEBP": ".webp"}
    ext = ext_map[fmt]

    pil_img = Image.fromarray(image)
    kw: dict = {}
    if fmt == "JPEG":
        if pil_img.mode in ("RGBA", "LA", "P"):
            pil_img = pil_img.convert("RGB")
        kw = {"quality": quality, "optimize": True}
    elif fmt == "PNG":
        kw = {"optimize": True}
    elif fmt == "TIFF":
        kw = {"compression": "lzw"}
    elif fmt == "WEBP":
        kw = {"quality": quality}

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        out_path = tmp.name
    pil_img.save(out_path, format=fmt, **kw)

    converted = np.array(Image.open(out_path).convert("RGB"))
    size = os.path.getsize(out_path)

    def _sz(b: int) -> str:
        return f"{b / 1024:.1f} KB" if b < 1024 * 1024 else f"{b / 1024 / 1024:.2f} MB"

    info = f"Format: {fmt}  |  File size: {_sz(size)}"
    return converted, info, out_path


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Chromatic Revival") as demo:
    gr.Markdown(
        "# Chromatic Revival\n"
        "AI-powered image processing — colorization, OCR, and format conversion. "
        "All runs locally, no API keys required."
    )

    with gr.Tab("Colorize"):
        gr.Markdown("Convert a black-and-white photograph to colour using the **Zhang et al. (2016)** Caffe model.")
        with gr.Row():
            col_in = gr.Image(label="Input (grayscale)", type="numpy")
            col_out = gr.Image(label="Colorized", type="numpy")
        col_btn = gr.Button("Colorize", variant="primary")
        col_metrics = gr.Textbox(label="Quality Metrics", interactive=False)
        col_btn.click(colorize, inputs=col_in, outputs=[col_out, col_metrics])

    with gr.Tab("Extract Text (OCR)"):
        gr.Markdown(
            "Local OCR powered by **docTR** (DBNet detection + CRNN recognition). "
            "No cloud APIs — everything runs on-device."
        )
        with gr.Row():
            ocr_in = gr.Image(label="Input Image", type="numpy")
            ocr_enhanced = gr.Image(label="Enhanced (CLAHE)", type="numpy")
        ocr_btn = gr.Button("Extract Text", variant="primary")
        ocr_text = gr.Textbox(label="Extracted Text (selectable)", lines=10, interactive=True)
        ocr_json = gr.Textbox(label="Structured JSON", lines=6, interactive=False)
        ocr_btn.click(extract_text, inputs=ocr_in, outputs=[ocr_enhanced, ocr_text, ocr_json])

    with gr.Tab("Convert Format"):
        gr.Markdown("Reformat images between **JPEG, PNG, BMP, TIFF, and WEBP**.")
        with gr.Row():
            conv_in = gr.Image(label="Input Image", type="numpy")
            conv_out = gr.Image(label="Preview", type="numpy")
        with gr.Row():
            fmt_dd = gr.Dropdown(
                choices=["JPEG", "PNG", "BMP", "TIFF", "WEBP"],
                value="JPEG", label="Target Format"
            )
            quality_sl = gr.Slider(1, 100, value=90, step=1, label="Quality (JPEG / WEBP only)")
        conv_btn = gr.Button("Convert & Download", variant="primary")
        conv_info = gr.Textbox(label="File Info", interactive=False)
        conv_file = gr.File(label="Download Converted File")
        conv_btn.click(
            convert_format,
            inputs=[conv_in, fmt_dd, quality_sl],
            outputs=[conv_out, conv_info, conv_file],
        )

if __name__ == "__main__":
    demo.launch()
