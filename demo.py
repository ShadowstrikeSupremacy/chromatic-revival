#!/usr/bin/env python3
"""
Advanced Image Colorization System - Demo Script
Academic Deep Learning Project

This script demonstrates all features of the image colorization system
including single image processing, batch processing, quality evaluation,
and comparison visualization.

Author: [Your Name]
Date: [Current Date]
Institution: [Your College]
"""

import cv2 as cv
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Import our modules
from src.colorizer import ImageColorizer
from src.utils import (
    validate_image_path, 
    get_image_files_from_directory,
    create_output_directory,
    save_comparison_image,
    calculate_image_statistics,
    format_time
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_single_image_colorization():
    """Demonstrate single image colorization."""
    print("\n" + "="*60)
    print("DEMO: Single Image Colorization")
    print("="*60)
    
    # Initialize colorizer
    try:
        colorizer = ImageColorizer()
        print("✓ Colorizer initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize colorizer: {e}")
        return
    
    # Test with available images
    test_images = ["new.jpg", "image.jpg", "input.png"]
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            continue
            
        print(f"\nProcessing: {image_path}")
        
        # Load image
        image = cv.imread(image_path)
        if image is None:
            print(f"✗ Could not load image: {image_path}")
            continue
        
        # Display image statistics
        stats = calculate_image_statistics(image)
        print(f"  Image size: {stats['shape']}")
        print(f"  Data type: {stats['dtype']}")
        print(f"  Value range: [{stats['min_value']:.1f}, {stats['max_value']:.1f}]")
        print(f"  Mean value: {stats['mean_value']:.1f}")
        
        # Colorize image
        start_time = time.time()
        try:
            colorized = colorizer.colorize_image(image)
            processing_time = time.time() - start_time
            
            print(f"✓ Colorization completed in {format_time(processing_time)}")
            
            # Save result
            output_path = f"demo_colorized_{Path(image_path).stem}.png"
            cv.imwrite(output_path, colorized)
            print(f"✓ Saved colorized image: {output_path}")
            
            # Evaluate quality
            metrics = colorizer.evaluate_colorization(image, colorized)
            print(f"  Quality Metrics:")
            print(f"    SSIM: {metrics['ssim']:.4f}")
            print(f"    PSNR: {metrics['psnr']:.2f} dB")
            print(f"    Colorfulness: {metrics['colorfulness']:.2f}")
            
            # Create comparison
            comparison_path = f"demo_comparison_{Path(image_path).stem}.png"
            save_comparison_image(image, colorized, comparison_path)
            print(f"✓ Saved comparison: {comparison_path}")
            
        except Exception as e:
            print(f"✗ Colorization failed: {e}")
    
    print("\n✓ Single image colorization demo completed")


def demo_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n" + "="*60)
    print("DEMO: Batch Processing")
    print("="*60)
    
    # Initialize colorizer
    try:
        colorizer = ImageColorizer()
        print("✓ Colorizer initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize colorizer: {e}")
        return
    
    # Get all available images
    current_dir = Path(".")
    image_files = []
    
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(current_dir.glob(f"*{ext}"))
    
    if not image_files:
        print("✗ No image files found for batch processing")
        return
    
    print(f"Found {len(image_files)} images for batch processing")
    
    # Create output directory
    output_dir = create_output_directory("demo_batch_output")
    
    # Process images
    image_paths = [str(f) for f in image_files]
    
    start_time = time.time()
    results = colorizer.batch_colorize(image_paths, output_dir)
    total_time = time.time() - start_time
    
    print(f"\nBatch Processing Results:")
    print(f"  Total images: {len(image_paths)}")
    print(f"  Successfully processed: {results['processed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Total processing time: {format_time(total_time)}")
    print(f"  Average time per image: {format_time(results['total_time']/results['processed'])}")
    print(f"  Output directory: {output_dir}")
    
    if results['errors']:
        print(f"  Errors encountered:")
        for error in results['errors']:
            print(f"    - {error}")
    
    print("\n✓ Batch processing demo completed")


def demo_quality_evaluation():
    """Demonstrate quality evaluation features."""
    print("\n" + "="*60)
    print("DEMO: Quality Evaluation")
    print("="*60)
    
    # Initialize colorizer
    try:
        colorizer = ImageColorizer()
        print("✓ Colorizer initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize colorizer: {e}")
        return
    
    # Test with a sample image
    test_image = "new.jpg"
    if not os.path.exists(test_image):
        print(f"✗ Test image not found: {test_image}")
        return
    
    # Load and process image
    image = cv.imread(test_image)
    if image is None:
        print(f"✗ Could not load test image: {test_image}")
        return
    
    colorized = colorizer.colorize_image(image)
    
    # Comprehensive evaluation
    print(f"\nQuality Evaluation for: {test_image}")
    
    # Calculate metrics
    metrics = colorizer.evaluate_colorization(image, colorized)
    
    print(f"\nEvaluation Metrics:")
    print(f"  Structural Similarity Index (SSIM): {metrics['ssim']:.4f}")
    print(f"    - Range: 0 to 1 (higher is better)")
    print(f"    - Interpretation: {get_ssim_interpretation(metrics['ssim'])}")
    
    print(f"  Peak Signal-to-Noise Ratio (PSNR): {metrics['psnr']:.2f} dB")
    print(f"    - Range: Typically 20-40 dB (higher is better)")
    print(f"    - Interpretation: {get_psnr_interpretation(metrics['psnr'])}")
    
    print(f"  Colorfulness Score: {metrics['colorfulness']:.2f}")
    print(f"    - Interpretation: {get_colorfulness_interpretation(metrics['colorfulness'])}")
    
    # Create detailed comparison plot
    create_detailed_comparison(image, colorized, "demo_detailed_comparison.png")
    print(f"\n✓ Detailed comparison saved: demo_detailed_comparison.png")
    
    print("\n✓ Quality evaluation demo completed")


def get_ssim_interpretation(ssim_value):
    """Get interpretation of SSIM value."""
    if ssim_value >= 0.95:
        return "Excellent quality"
    elif ssim_value >= 0.90:
        return "Very good quality"
    elif ssim_value >= 0.80:
        return "Good quality"
    elif ssim_value >= 0.70:
        return "Fair quality"
    else:
        return "Poor quality"


def get_psnr_interpretation(psnr_value):
    """Get interpretation of PSNR value."""
    if psnr_value >= 40:
        return "Excellent quality"
    elif psnr_value >= 30:
        return "Good quality"
    elif psnr_value >= 20:
        return "Acceptable quality"
    else:
        return "Poor quality"


def get_colorfulness_interpretation(colorfulness_value):
    """Get interpretation of colorfulness value."""
    if colorfulness_value >= 50:
        return "Very vibrant colors"
    elif colorfulness_value >= 30:
        return "Moderately vibrant colors"
    elif colorfulness_value >= 15:
        return "Somewhat muted colors"
    else:
        return "Very muted colors"


def create_detailed_comparison(original, colorized, output_path):
    """Create a detailed comparison with multiple views."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    original_rgb = cv.cvtColor(original, cv.COLOR_BGR2RGB)
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Colorized image
    colorized_rgb = cv.cvtColor(colorized, cv.COLOR_BGR2RGB)
    axes[0, 1].imshow(colorized_rgb)
    axes[0, 1].set_title('Colorized Image')
    axes[0, 1].axis('off')
    
    # Grayscale comparison
    original_gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
    colorized_gray = cv.cvtColor(colorized, cv.COLOR_BGR2GRAY)
    
    axes[0, 2].imshow(original_gray, cmap='gray')
    axes[0, 2].set_title('Original (Grayscale)')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(colorized_gray, cmap='gray')
    axes[1, 0].set_title('Colorized (Grayscale)')
    axes[1, 0].axis('off')
    
    # Histogram comparison
    axes[1, 1].hist(original_gray.ravel(), bins=256, alpha=0.7, label='Original', color='blue')
    axes[1, 1].hist(colorized_gray.ravel(), bins=256, alpha=0.7, label='Colorized', color='red')
    axes[1, 1].set_title('Grayscale Histogram Comparison')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    
    # Color channels
    colorized_lab = cv.cvtColor(colorized, cv.COLOR_BGR2Lab)
    axes[1, 2].imshow(colorized_lab[:, :, 1], cmap='RdBu')
    axes[1, 2].set_title('Colorized (a-channel)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def demo_performance_analysis():
    """Demonstrate performance analysis features."""
    print("\n" + "="*60)
    print("DEMO: Performance Analysis")
    print("="*60)
    
    # Initialize colorizer
    try:
        colorizer = ImageColorizer()
        print("✓ Colorizer initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize colorizer: {e}")
        return
    
    # Test with different image sizes
    test_image = "new.jpg"
    if not os.path.exists(test_image):
        print(f"✗ Test image not found: {test_image}")
        return
    
    original_image = cv.imread(test_image)
    if original_image is None:
        print(f"✗ Could not load test image: {test_image}")
        return
    
    # Test different image sizes
    sizes = [(224, 224), (512, 512), (1024, 1024)]
    results = []
    
    print(f"\nPerformance Analysis with Different Image Sizes:")
    print(f"{'Size':<15} {'Time (s)':<12} {'Memory (MB)':<15}")
    print("-" * 45)
    
    for size in sizes:
        # Resize image
        resized = cv.resize(original_image, size)
        
        # Measure time
        start_time = time.time()
        colorized = colorizer.colorize_image(resized)
        processing_time = time.time() - start_time
        
        # Estimate memory usage (rough calculation)
        memory_usage = (resized.nbytes + colorized.nbytes) / (1024 * 1024)
        
        results.append({
            'size': size,
            'time': processing_time,
            'memory': memory_usage
        })
        
        print(f"{size[0]}x{size[1]:<10} {processing_time:<12.3f} {memory_usage:<15.2f}")
    
    # Calculate efficiency metrics
    print(f"\nEfficiency Analysis:")
    for i, result in enumerate(results):
        if i > 0:
            time_ratio = result['time'] / results[i-1]['time']
            size_ratio = (result['size'][0] * result['size'][1]) / (results[i-1]['size'][0] * results[i-1]['size'][1])
            efficiency = size_ratio / time_ratio
            print(f"  {result['size'][0]}x{result['size'][1]}: {efficiency:.2f}x efficiency vs previous size")
    
    print("\n✓ Performance analysis demo completed")


def main():
    """Run all demos."""
    print("Advanced Image Colorization System - Demo")
    print("Academic Deep Learning Project")
    print("="*60)
    
    # Check if model files exist
    if not os.path.exists("models/colorization_deploy_v2.prototxt"):
        print("✗ Model files not found. Please ensure all model files are present.")
        print("Required files:")
        print("  - models/colorization_deploy_v2.prototxt")
        print("  - models/colorization_release_v2.caffemodel")
        print("  - pts_in_hull.npy")
        return
    
    if not os.path.exists("pts_in_hull.npy"):
        print("✗ pts_in_hull.npy not found in current directory.")
        return
    
    # Run all demos
    try:
        demo_single_image_colorization()
        demo_batch_processing()
        demo_quality_evaluation()
        demo_performance_analysis()
        
        print("\n" + "="*60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        print("  - demo_colorized_*.png (colorized images)")
        print("  - demo_comparison_*.png (before/after comparisons)")
        print("  - demo_detailed_comparison.png (detailed analysis)")
        print("  - demo_batch_output/ (batch processing results)")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        logger.error(f"Demo error: {e}")


if __name__ == "__main__":
    main() 