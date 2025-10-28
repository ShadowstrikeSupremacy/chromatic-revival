"""
Image Colorization Module

This module handles the core colorization logic using a pre-trained Caffe model.
It converts black and white images to color by predicting the a and b channels
from the L channel in Lab color space.

Author: Siddh
Date: December 2024
"""

import cv2 as cv
import numpy as np
import os
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageColorizer:
    """
    Image Colorization Class
    
    This class handles the colorization of black and white images using a pre-trained
    Caffe model. It works in Lab color space where L is brightness and a,b are colors.
    
    Attributes:
        model_path (str): Path to the Caffe model file
        prototxt_path (str): Path to the prototxt file
        pts_path (str): Path to the cluster centers file
        net: Loaded Caffe neural network
        pts_in_hull: Cluster centers for colorization
        input_size (tuple): Input size for the neural network
    """
    
    def __init__(self, model_dir: str = "./models"):
        """
        Initialize the ImageColorizer with model files.
        
        Args:
            model_dir (str): Directory containing model files
        """
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "colorization_release_v2.caffemodel"
        self.prototxt_path = self.model_dir / "colorization_deploy_v2.prototxt"
        self.pts_path = Path("./pts_in_hull.npy")
        
        self.input_size = (224, 224)
        self.net = None
        self.pts_in_hull = None
        
        self._load_model()
        logger.info("ImageColorizer initialized successfully")
    
    def _load_model(self):
        """Load the Caffe model and cluster centers."""
        try:
            # Check if model files exist
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            if not self.prototxt_path.exists():
                raise FileNotFoundError(f"Prototxt file not found: {self.prototxt_path}")
            if not self.pts_path.exists():
                raise FileNotFoundError(f"Cluster centers file not found: {self.pts_path}")
            
            # Load the neural network
            self.net = cv.dnn.readNetFromCaffe(str(self.prototxt_path), str(self.model_path))
            
            # Load cluster centers
            self.pts_in_hull = np.load(str(self.pts_path))
            self.pts_in_hull = self.pts_in_hull.transpose().reshape(2, 313, 1, 1)
            
            # Set up the network layers
            self.net.getLayer(self.net.getLayerId('class8_ab')).blobs = [
                self.pts_in_hull.astype(np.float32)
            ]
            self.net.getLayer(self.net.getLayerId('conv8_313_rh')).blobs = [
                np.full([1, 313], 2.606, np.float32)
            ]
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the input image for colorization.
        
        Args:
            image (np.ndarray): Input BGR image
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Preprocessed L channel and original image
        """
        # Convert BGR to RGB (OpenCV uses BGR by default)
        rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] - this helps with the neural network
        rgb_img = (rgb_img * 1.0 / 255).astype(np.float32)
        
        # Convert to Lab color space - this is where the magic happens
        lab_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2Lab)
        
        # Extract L channel (the brightness part)
        l_channel = lab_img[:, :, 0]
        
        # Resize L channel for network input and center it
        l_channel_resize = cv.resize(l_channel, self.input_size)
        l_channel_resize -= 50  # Center the L channel values
        
        return l_channel_resize, rgb_img
    
    def colorize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Colorize a black and white image.
        
        Args:
            image (np.ndarray): Input BGR image
            
        Returns:
            np.ndarray: Colorized image in BGR format
        """
        start_time = time.time()
        
        try:
            # Preprocess the image
            l_channel_resize, rgb_img = self.preprocess_image(image)
            
            # Create blob for network input - this is what the neural network expects
            blob = cv.dnn.blobFromImage(l_channel_resize)
            self.net.setInput(blob)
            
            # Forward pass - this is where the actual colorization happens
            ab_channel = self.net.forward()[0, :, :, :].transpose((1, 2, 0))
            
            # Resize ab channel to original image size
            original_height, original_width = rgb_img.shape[:2]
            ab_channel_us = cv.resize(ab_channel, (original_width, original_height))
            
            # Combine L and ab channels - put it all back together
            l_channel = cv.cvtColor(rgb_img, cv.COLOR_RGB2Lab)[:, :, 0]
            lab_output = np.concatenate((l_channel[:, :, np.newaxis], ab_channel_us), axis=2)
            
            # Convert back to BGR and clip values to valid range
            bgr_output = np.clip(cv.cvtColor(lab_output, cv.COLOR_Lab2BGR), 0, 1)
            
            processing_time = time.time() - start_time
            logger.info(f"Image colorized successfully in {processing_time:.2f} seconds")
            
            return (bgr_output * 255).astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error during colorization: {str(e)}")
            raise
    
    def batch_colorize(self, image_paths: list, output_dir: str = "./output") -> Dict[str, Any]:
        """
        Colorize multiple images in batch.
        
        Args:
            image_paths (list): List of image file paths
            output_dir (str): Output directory for colorized images
            
        Returns:
            Dict[str, Any]: Processing statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = {
            'processed': 0,
            'failed': 0,
            'total_time': 0,
            'errors': []
        }
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                
                # Load image
                image = cv.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not load image: {image_path}")
                
                # Colorize image
                start_time = time.time()
                colorized = self.colorize_image(image)
                processing_time = time.time() - start_time
                
                # Save result
                output_filename = f"colorized_{Path(image_path).stem}.png"
                output_path_full = output_path / output_filename
                cv.imwrite(str(output_path_full), colorized)
                
                results['processed'] += 1
                results['total_time'] += processing_time
                
                logger.info(f"Saved colorized image: {output_path_full}")
                
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"{image_path}: {str(e)}")
                logger.error(f"Failed to process {image_path}: {str(e)}")
        
        logger.info(f"Batch processing completed. Processed: {results['processed']}, Failed: {results['failed']}")
        return results
    
    def evaluate_colorization(self, original: np.ndarray, colorized: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the quality of colorization using various metrics.
        
        Args:
            original (np.ndarray): Original image
            colorized (np.ndarray): Colorized image
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        try:
            # Ensure both images have the same size
            if original.shape != colorized.shape:
                colorized = cv.resize(colorized, (original.shape[1], original.shape[0]))

            # Calculate SSIM
            ssim_score = ssim(original, colorized, channel_axis=2, data_range=255)
            
            # Calculate PSNR
            psnr_score = psnr(original, colorized, data_range=255)
            
            # Calculate colorfulness using the ab channels
            lab_img = cv.cvtColor(colorized, cv.COLOR_BGR2Lab)
            a = lab_img[:,:,1]
            b = lab_img[:,:,2]
            a_std = np.std(a)
            b_std = np.std(b)
            a_mean = np.mean(a)
            b_mean = np.mean(b)
            colorfulness = np.sqrt((a_std ** 2 + b_std ** 2) + 0.3 * np.sqrt(a_mean ** 2 + b_mean ** 2))
            
            return {
                'ssim': float(ssim_score),
                'psnr': float(psnr_score),
                'colorfulness': float(colorfulness)
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def create_comparison_plot(self, original: np.ndarray, colorized: np.ndarray, 
                              save_path: Optional[str] = None) -> None:
        """
        Create a comparison plot of original vs colorized image.
        
        Args:
            original (np.ndarray): Original image
            colorized (np.ndarray): Colorized image
            save_path (Optional[str]): Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        original_rgb = cv.cvtColor(original, cv.COLOR_BGR2RGB)
        axes[0].imshow(original_rgb)
        axes[0].set_title('Original (Grayscale)')
        axes[0].axis('off')
        
        # Colorized image
        colorized_rgb = cv.cvtColor(colorized, cv.COLOR_BGR2RGB)
        axes[1].imshow(colorized_rgb)
        axes[1].set_title('Colorized')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved: {save_path}")
        
        plt.show()


def main():
    """Example usage of the ImageColorizer class."""
    try:
        # Initialize colorizer
        colorizer = ImageColorizer()
        
        # Load and colorize a single image
        image_path = "new.jpg"
        if os.path.exists(image_path):
            image = cv.imread(image_path)
            colorized = colorizer.colorize_image(image)
            
            # Save result
            cv.imwrite("result.png", colorized)
            
            # Create comparison plot
            colorizer.create_comparison_plot(image, colorized, "comparison.png")
            
            # Evaluate quality
            metrics = colorizer.evaluate_colorization(image, colorized)
            print(f"Evaluation metrics: {metrics}")
            
        else:
            logger.warning(f"Image file not found: {image_path}")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main() 
