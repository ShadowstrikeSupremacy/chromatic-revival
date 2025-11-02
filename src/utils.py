"""
Utility functions for Advanced Image Colorization System
Academic Deep Learning Project

This module provides utility functions for image processing, validation,
and various helper functions used throughout the colorization system.
"""

import cv2 as cv
import numpy as np
import os
from pathlib import Path
from typing import Tuple, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def validate_image_path(image_path: str) -> bool:
    """
    Validate if the given path points to a valid image file.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        bool: True if valid image file, False otherwise
    """
    if not os.path.exists(image_path):
        logger.error(f"Image file does not exist: {image_path}")
        return False
    
    # Check file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    file_extension = Path(image_path).suffix.lower()
    
    if file_extension not in valid_extensions:
        logger.error(f"Unsupported file format: {file_extension}")
        return False
    
    # Try to read the image
    try:
        image = cv.imread(image_path)
        if image is None:
            logger.error(f"Could not read image file: {image_path}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error reading image file: {str(e)}")
        return False


def is_grayscale(image: np.ndarray) -> bool:
    """
    Check if an image is grayscale.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        bool: True if grayscale, False otherwise
    """
    if len(image.shape) == 2:
        return True
    elif len(image.shape) == 3:
        # Check if all channels are equal (grayscale)
        return np.allclose(image[:, :, 0], image[:, :, 1]) and np.allclose(image[:, :, 1], image[:, :, 2])
    return False


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale if it's not already.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        np.ndarray: Grayscale image
    """
    if is_grayscale(image):
        return image
    
    if len(image.shape) == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    return image


def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                maintain_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resize an image to target size.
    
    Args:
        image (np.ndarray): Input image
        target_size (Tuple[int, int]): Target width and height
        maintain_aspect_ratio (bool): Whether to maintain aspect ratio
        
    Returns:
        np.ndarray: Resized image
    """
    if maintain_aspect_ratio:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        return cv.resize(image, (new_w, new_h))
    else:
        return cv.resize(image, target_size)


def enhance_image_quality(image: np.ndarray) -> np.ndarray:
    """
    Apply basic image enhancement techniques.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        np.ndarray: Enhanced image
    """
    # Convert to float for processing
    image_float = image.astype(np.float32) / 255.0
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if len(image_float.shape) == 3:
        lab = cv.cvtColor(image_float, cv.COLOR_BGR2Lab)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply((lab[:, :, 0] * 255).astype(np.uint8)) / 255.0
        enhanced = cv.cvtColor(lab, cv.COLOR_Lab2BGR)
    else:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply((image_float * 255).astype(np.uint8)) / 255.0
    
    return (enhanced * 255).astype(np.uint8)


def calculate_image_statistics(image: np.ndarray) -> dict:
    """
    Calculate various statistics for an image.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        dict: Dictionary containing image statistics
    """
    stats = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min_value': float(np.min(image)),
        'max_value': float(np.max(image)),
        'mean_value': float(np.mean(image)),
        'std_value': float(np.std(image)),
        'is_grayscale': is_grayscale(image)
    }
    
    if len(image.shape) == 3:
        # Color image statistics
        for i, channel in enumerate(['B', 'G', 'R']):
            stats[f'{channel}_mean'] = float(np.mean(image[:, :, i]))
            stats[f'{channel}_std'] = float(np.std(image[:, :, i]))
    
    return stats


def create_image_grid(images: List[np.ndarray], titles: Optional[List[str]] = None,
                     grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Create a grid of images for comparison.
    
    Args:
        images (List[np.ndarray]): List of images to display
        titles (Optional[List[str]]): List of titles for each image
        grid_size (Optional[Tuple[int, int]]): Grid size (rows, cols)
        
    Returns:
        np.ndarray: Grid image
    """
    if not images:
        raise ValueError("No images provided")
    
    n_images = len(images)
    
    if grid_size is None:
        # Auto-calculate grid size
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size
    
    # Get dimensions from first image
    img_height, img_width = images[0].shape[:2]
    
    # Create grid
    grid_height = rows * img_height
    grid_width = cols * img_width
    
    if len(images[0].shape) == 3:
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    else:
        grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
    
    # Place images in grid
    for i, image in enumerate(images):
        if i >= rows * cols:
            break
        
        row = i // cols
        col = i % cols
        
        y_start = row * img_height
        y_end = y_start + img_height
        x_start = col * img_width
        x_end = x_start + img_width
        
        # Resize image to fit grid cell
        resized_image = resize_image(image, (img_width, img_height))
        
        if len(resized_image.shape) == 2 and len(grid.shape) == 3:
            # Convert grayscale to BGR for grid
            resized_image = cv.cvtColor(resized_image, cv.COLOR_GRAY2BGR)
        
        grid[y_start:y_end, x_start:x_end] = resized_image
    
    return grid


def save_comparison_image(original: np.ndarray, colorized: np.ndarray, 
                         output_path: str, title: str = "Image Colorization Comparison") -> None:
    """
    Save a side-by-side comparison of original and colorized images.
    
    Args:
        original (np.ndarray): Original image
        colorized (np.ndarray): Colorized image
        output_path (str): Path to save the comparison image
        title (str): Title for the comparison
    """
    # Ensure both images are the same size
    if original.shape != colorized.shape:
        colorized = resize_image(colorized, (original.shape[1], original.shape[0]))
    
    # Create comparison image
    comparison = create_image_grid([original, colorized], 
                                 titles=["Original", "Colorized"])
    
    # Save the comparison
    cv.imwrite(output_path, comparison)
    logger.info(f"Comparison image saved: {output_path}")


def get_image_files_from_directory(directory: str, 
                                 extensions: Optional[List[str]] = None) -> List[str]:
    """
    Get all image files from a directory.
    
    Args:
        directory (str): Directory path
        extensions (Optional[List[str]]): List of file extensions to include
        
    Returns:
        List[str]: List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    image_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return image_files
    
    for file_path in directory_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            image_files.append(str(file_path))
    
    return sorted(image_files)


def create_output_directory(output_dir: str) -> str:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir (str): Output directory path
        
    Returns:
        str: Created directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created/verified: {output_path}")
    return str(output_path)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable format.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def validate_model_files(model_dir: str) -> bool:
    """
    Validate that all required model files are present.
    
    Args:
        model_dir (str): Directory containing model files
        
    Returns:
        bool: True if all files are present, False otherwise
    """
    required_files = [
        "colorization_deploy_v2.prototxt",
        "colorization_release_v2.caffemodel"
    ]
    
    model_path = Path(model_dir)
    
    for file_name in required_files:
        file_path = model_path / file_name
        if not file_path.exists():
            logger.error(f"Required model file missing: {file_path}")
            return False
    
    # Check for pts_in_hull.npy in current directory
    pts_file = Path("pts_in_hull.npy")
    if not pts_file.exists():
        logger.error(f"Required file missing: {pts_file}")
        return False
    
    logger.info("All required model files are present")
    return True


def get_system_info() -> dict:
    """
    Get system information for debugging and logging.
    
    Returns:
        dict: System information
    """
    import platform
    import sys
    
    info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'opencv_version': cv.__version__,
        'numpy_version': np.__version__
    }
    
    return info 