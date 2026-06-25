"""
Standalone OCR Pipeline Module

Two-stage local OCR: image enhancement followed by neural-network text
extraction via docTR (DBNet detection + CRNN recognition). No cloud APIs.

Performance note: docTR's db_resnet50 + crnn_vgg16_bn combination matches
or exceeds GoogleVision / AWS Textract on standard document benchmarks.

Example:
    >>> pipeline = StandaloneOcrPipeline()
    >>> img = pipeline.optimize_image_for_ocr("scan.png")
    >>> results = pipeline.extract_text(img)
    >>> for r in results:
    ...     print(r["text"], r["confidence"])
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class StandaloneOcrPipeline:
    """Two-stage OCR pipeline using docTR.

    Stage 1 — ``optimize_image_for_ocr``: CLAHE contrast enhancement
    prepares the image for the detection network while preserving colour
    information that the recognition model uses.

    Stage 2 — ``extract_text``: passes the enhanced image directly to the
    docTR predictor (DBNet for word localisation, CRNN for character
    recognition) and normalises the hierarchical result into a flat list of
    dicts that the rest of the application consumes.

    The docTR predictor returns words already arranged in reading order
    (Block → Line → Word), so no post-hoc sorting is required.

    Attributes:
        predictor: Initialised ``OCRPredictor`` instance from docTR.
    """

    def __init__(self, use_gpu: bool = False) -> None:
        """Load docTR pretrained models (DBNet + CRNN).

        Model weights (~100 MB total) are cached in ``~/.cache/doctr/models/``
        after the first download. Subsequent runs load from cache instantly.

        Args:
            use_gpu: Pass ``True`` to run inference on CUDA GPU. Falls back
                to CPU automatically when CUDA is unavailable.
        """
        from doctr.models import ocr_predictor

        logger.info("Loading docTR OCR predictor (db_resnet50 + crnn_vgg16_bn) ...")
        self.predictor = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="crnn_vgg16_bn",
            pretrained=True,
            assume_straight_pages=True,  # faster path for upright document pages
        )

        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self.predictor = self.predictor.cuda()
                    logger.info("docTR running on GPU")
                else:
                    logger.warning("use_gpu=True but CUDA is not available — using CPU")
            except ImportError:
                logger.warning("PyTorch not found — cannot move predictor to GPU")

        logger.info("docTR predictor ready")

    # ------------------------------------------------------------------
    # Stage 1 — Image enhancement
    # ------------------------------------------------------------------

    def optimize_image_for_ocr(self, image_path: str) -> np.ndarray:
        """Load an image and apply CLAHE contrast enhancement.

        Unlike hard binarisation, CLAHE (Contrast Limited Adaptive Histogram
        Equalisation) boosts local contrast while keeping the image in full
        colour. docTR's recognition model uses colour cues, so preserving them
        improves accuracy on real-world scans.

        Enhancement steps:

        1. Load as BGR, validate.
        2. Convert to LAB colour space.
        3. Apply CLAHE on the L (luminance) channel only — colour channels
           are untouched so hue information is preserved.
        4. Convert back to BGR then to RGB for display and inference.

        Args:
            image_path: Filesystem path to the input image.

        Returns:
            Enhanced ``(H, W, 3)`` ``uint8`` RGB array ready to pass to
            :meth:`extract_text` or display in the GUI.

        Raises:
            FileNotFoundError: If ``image_path`` does not exist.
            ValueError: If OpenCV cannot decode the file.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        bgr = cv.imread(str(path))
        if bgr is None:
            raise ValueError(
                f"OpenCV could not decode the image at: {image_path}. "
                "Ensure the file is a valid image format."
            )

        logger.debug(f"Loaded image {bgr.shape} from {image_path}")

        # CLAHE on the luminance channel preserves colour while boosting contrast
        lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)

        # Return as RGB — PIL and docTR both expect RGB
        return cv.cvtColor(enhanced_bgr, cv.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # Stage 2 — Text extraction
    # ------------------------------------------------------------------

    def extract_text(self, processed_img: np.ndarray) -> list[dict]:
        """Run docTR on a preprocessed image and return structured results.

        The docTR predictor accepts a list of ``(H, W, 3)`` uint8 RGB arrays
        directly — no intermediate file I/O needed. The hierarchical result
        (Document → Page → Block → Line → Word) is flattened into a list of
        dicts that maintains reading order.

        Word bounding boxes from docTR are normalised to [0, 1]; this method
        converts them to absolute pixel coordinates.

        Args:
            processed_img: Enhanced image as a NumPy array — typically the
                output of :meth:`optimize_image_for_ocr`. Must be
                ``(H, W, 3)`` uint8 RGB. Greyscale input is auto-converted.

        Returns:
            List of dicts in reading order (top-left to bottom-right), schema::

                {
                    "text":       str,               # recognised word
                    "confidence": float,             # model confidence [0, 1]
                    "bbox":       [x1, y1, x2, y2]  # absolute pixel coords
                }

            Returns an empty list when no text is detected.
        """
        # Normalise input: docTR expects (H, W, 3) uint8 RGB
        if processed_img.ndim == 2:
            img = cv.cvtColor(processed_img, cv.COLOR_GRAY2RGB)
        elif processed_img.shape[2] == 4:
            img = cv.cvtColor(processed_img, cv.COLOR_RGBA2RGB)
        else:
            img = processed_img

        logger.info("Running docTR text detection + recognition ...")
        # Predictor accepts a list of numpy arrays — no DocumentFile needed
        result = self.predictor([img])

        output: list[dict] = []
        for page in result.pages:
            h, w = page.dimensions  # (height, width) matching img.shape[:2]
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        # geometry: ((x_min, y_min), (x_max, y_max)) in [0, 1]
                        (x1n, y1n), (x2n, y2n) = word.geometry
                        output.append(
                            {
                                "text": word.value,
                                "confidence": float(word.confidence),
                                "bbox": [
                                    int(x1n * w), int(y1n * h),
                                    int(x2n * w), int(y2n * h),
                                ],
                            }
                        )

        logger.info(f"Extracted {len(output)} word(s)")
        return output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.ocr",
        description=(
            "Standalone local OCR via docTR — no cloud dependencies. "
            "Powered by DBNet (detection) + CRNN (recognition)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.ocr --image scan.png
  python -m src.ocr --image document.jpg --output results.json
  python src/ocr.py --image receipt.png --output receipt.json --gpu
        """,
    )
    parser.add_argument("--image", required=True, metavar="PATH",
                        help="Path to the input image.")
    parser.add_argument("--output", metavar="PATH", default=None,
                        help="Save structured JSON results to this path.")
    parser.add_argument("--gpu", action="store_true", default=False,
                        help="Enable GPU acceleration (requires CUDA PyTorch).")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    pipeline = StandaloneOcrPipeline(use_gpu=args.gpu)

    print(f"\nEnhancing: {args.image}")
    img = pipeline.optimize_image_for_ocr(args.image)

    print("Running detection + recognition ...")
    results = pipeline.extract_text(img)

    if not results:
        print("\nNo text detected.")
        sys.exit(0)

    print(f"\nExtracted {len(results)} word(s):\n")
    for i, item in enumerate(results, 1):
        print(f"  [{i:02d}]  \"{item['text']}\"")
        print(f"         confidence : {item['confidence']:.1%}")
        print(f"         bbox       : {item['bbox']}")
        print()

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print(f"Results saved to: {out}")
