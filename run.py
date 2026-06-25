#!/usr/bin/env python3
"""
Image Colorization Launcher

Simple script to run different parts of the colorization system.
"""

import sys
import os
import argparse
from pathlib import Path

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        description="Image Colorization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py gui                              # Launch the GUI
  python run.py colorize input.jpg              # Colorize a single image
  python run.py batch input_folder              # Batch process a folder
  python run.py ocr input.jpg                   # Extract text from an image
  python run.py ocr input.jpg -o results.json   # Save OCR results to JSON
  python run.py convert input.png --to jpeg     # Convert PNG to JPEG
  python run.py convert input.png --to webp -o out.webp
        """
    )

    parser.add_argument(
        'mode',
        choices=['gui', 'demo', 'colorize', 'batch', 'ocr', 'convert'],
        help='Mode to run the system in'
    )

    parser.add_argument(
        'input',
        nargs='?',
        help='Input image or folder path'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output path for the result file'
    )
    parser.add_argument(
        '--to',
        choices=['jpeg', 'png', 'bmp', 'tiff', 'webp'],
        default='jpeg',
        help='Target format for convert mode (default: jpeg)'
    )
    parser.add_argument(
        '--quality', type=int, default=90, metavar='1-100',
        help='JPEG/WEBP quality 1-100 (default: 90)'
    )
    
    args = parser.parse_args()
    
    # Check if model files exist
    if not os.path.exists("models/colorization_deploy_v2.prototxt"):
        print("✗ Model files not found!")
        print("Please ensure the following files are present:")
        print("  - models/colorization_deploy_v2.prototxt")
        print("  - models/colorization_release_v2.caffemodel")
        print("  - pts_in_hull.npy")
        return 1
    
    if not os.path.exists("pts_in_hull.npy"):
        print("✗ pts_in_hull.npy not found in current directory!")
        return 1
    
    # Run the selected mode
    if args.mode == 'gui':
        print("Launching GUI...")
        try:
            from src.gui import main as gui_main
            gui_main()
        except ImportError as e:
            print(f"✗ Failed to import GUI: {e}")
            return 1
    
    elif args.mode == 'demo':
        print("Running demo...")
        try:
            import demo
            demo.main()
        except ImportError as e:
            print(f"✗ Failed to import demo: {e}")
            return 1
    
    elif args.mode == 'colorize':
        if not args.input:
            print("✗ Please provide an input image path")
            return 1
        
        if not os.path.exists(args.input):
            print(f"✗ Input file not found: {args.input}")
            return 1
        
        print(f"Colorizing: {args.input}")
        try:
            from src.colorizer import ImageColorizer
            import cv2 as cv
            
            # Initialize colorizer
            colorizer = ImageColorizer()
            
            # Load and process image
            image = cv.imread(args.input)
            if image is None:
                print(f"✗ Could not load image: {args.input}")
                return 1
            
            # Colorize image
            colorized = colorizer.colorize_image(image)
            
            # Save result
            output_path = args.output or f"colorized_{Path(args.input).stem}.png"
            cv.imwrite(output_path, colorized)
            
            print(f"✓ Colorized image saved: {output_path}")
            
            # Show metrics
            metrics = colorizer.evaluate_colorization(image, colorized)
            print(f"Quality Metrics:")
            print(f"  SSIM: {metrics['ssim']:.4f}")
            print(f"  PSNR: {metrics['psnr']:.2f} dB")
            print(f"  Colorfulness: {metrics['colorfulness']:.2f}")
            
        except Exception as e:
            print(f"✗ Colorization failed: {e}")
            return 1
    
    elif args.mode == 'batch':
        if not args.input:
            print("✗ Please provide an input folder path")
            return 1
        
        if not os.path.exists(args.input):
            print(f"✗ Input folder not found: {args.input}")
            return 1
        
        print(f"Batch processing folder: {args.input}")
        try:
            from src.colorizer import ImageColorizer
            from src.utils import get_image_files_from_directory, create_output_directory
            
            # Initialize colorizer
            colorizer = ImageColorizer()
            
            # Get image files
            image_files = get_image_files_from_directory(args.input)
            if not image_files:
                print("✗ No image files found in the specified folder")
                return 1
            
            print(f"Found {len(image_files)} images to process")
            
            # Create output directory
            output_dir = args.output or "batch_output"
            create_output_directory(output_dir)
            
            # Process images
            results = colorizer.batch_colorize(image_files, output_dir)
            
            print(f"\nBatch processing completed!")
            print(f"  Processed: {results['processed']} images")
            print(f"  Failed: {results['failed']} images")
            print(f"  Output directory: {output_dir}")
            
            if results['errors']:
                print("Errors encountered:")
                for error in results['errors']:
                    print(f"  - {error}")
            
        except Exception as e:
            print(f"✗ Batch processing failed: {e}")
            return 1

    elif args.mode == 'ocr':
        if not args.input:
            print("✗ Please provide an input image path"); return 1
        if not os.path.exists(args.input):
            print(f"✗ Input file not found: {args.input}"); return 1
        print(f"Running OCR on: {args.input}")
        try:
            import json
            from src.ocr import StandaloneOcrPipeline
            pipeline = StandaloneOcrPipeline(use_gpu=False)
            print("  Enhancing image ...")
            processed = pipeline.optimize_image_for_ocr(args.input)
            print("  Running detection + recognition ...")
            results = pipeline.extract_text(processed)
            if not results:
                print("\nNo text detected."); return 0
            print(f"\nExtracted {len(results)} word(s):\n")
            for i, item in enumerate(results, 1):
                print(f"  [{i:02d}]  \"{item['text']}\"  ({item['confidence']:.1%})")
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as fh:
                    json.dump(results, fh, indent=2, ensure_ascii=False)
                print(f"\n✓ Results saved to: {args.output}")
        except Exception as e:
            print(f"✗ OCR failed: {e}"); return 1

    elif args.mode == 'convert':
        if not args.input:
            print("✗ Please provide an input image path"); return 1
        if not os.path.exists(args.input):
            print(f"✗ Input file not found: {args.input}"); return 1
        fmt = args.to.upper()
        ext_map = {'JPEG': '.jpg', 'PNG': '.png', 'BMP': '.bmp', 'TIFF': '.tif', 'WEBP': '.webp'}
        output_path = args.output or (Path(args.input).stem + ext_map[fmt])
        print(f"Converting: {args.input}  →  {fmt}  →  {output_path}")
        try:
            from PIL import Image as _Image
            img = _Image.open(args.input)
            kw: dict = {}
            if fmt == 'JPEG':
                if img.mode in ('RGBA', 'LA', 'P'): img = img.convert('RGB')
                kw = {'quality': args.quality, 'optimize': True}
            elif fmt == 'PNG':
                kw = {'optimize': True}
            elif fmt == 'TIFF':
                kw = {'compression': 'lzw'}
            elif fmt == 'WEBP':
                kw = {'quality': args.quality}
            img.save(output_path, format=fmt, **kw)
            ob, cb = os.path.getsize(args.input), os.path.getsize(output_path)
            def _sz(b): return f"{b/1024:.1f} KB" if b < 1024*1024 else f"{b/1024/1024:.2f} MB"
            print(f"✓ Saved: {output_path}")
            print(f"  Original : {_sz(ob)}  →  Converted: {_sz(cb)}  ({cb/ob*100:.1f}%)")
        except Exception as e:
            print(f"✗ Conversion failed: {e}"); return 1

    return 0


if __name__ == "__main__":
    sys.exit(main()) 