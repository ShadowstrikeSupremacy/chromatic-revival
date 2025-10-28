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
  python run.py gui                    # Launch the GUI
  python run.py demo                   # Run the demo script
  python run.py colorize input.jpg     # Colorize a single image
  python run.py batch input_folder     # Batch process a folder
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['gui', 'demo', 'colorize', 'batch'],
        help='Mode to run the system in'
    )
    
    parser.add_argument(
        'input',
        nargs='?',
        help='Input image or folder path'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        help='Output path for colorized images'
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
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 