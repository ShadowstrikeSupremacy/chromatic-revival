"""
Image Colorization GUI

This module provides a graphical user interface for the image colorization system.
It includes features like drag and drop, real-time feedback, and quality metrics.

Author: Siddh
Date: December 2024
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
import os
import threading
from pathlib import Path
import logging
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    from .colorizer import ImageColorizer
except ImportError:
    from colorizer import ImageColorizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModernColorizationGUI:
    """GUI for Image Colorization."""

    def __init__(self, root: tk.Tk):
        """Initialize the GUI."""
        self.root = root
        self.root.title("Advanced Image Colorization System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        # Initialize colorizer
        try:
            self.colorizer = ImageColorizer()
            self.model_loaded = True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.model_loaded = False

        # Variables
        self.original_image = None
        self.colorized_image = None
        self.original_path = None
        self.processing = False

        # Setup GUI
        self._setup_styles()
        self._create_widgets()
        self._setup_layout()

        logger.info("GUI initialized successfully")

    def _setup_styles(self):
        """Configure modern styling for the GUI."""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2c3e50')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), foreground='#34495e')
        style.configure('Info.TLabel', font=('Arial', 10), foreground='#7f8c8d')
        style.configure('Primary.TButton', font=('Arial', 10, 'bold'), background='#3498db', foreground='white')
        style.configure('Success.TButton', font=('Arial', 10, 'bold'), background='#27ae60', foreground='white')
        style.configure('Warning.TButton', font=('Arial', 10, 'bold'), background='#f39c12', foreground='white')

    def _create_widgets(self):
        """Create all GUI widgets."""
        self.main_frame = ttk.Frame(self.root, padding="10")

        self.title_label = ttk.Label(self.main_frame, text="Advanced Image Colorization System", style='Title.TLabel')
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding="10")

        self.file_frame = ttk.Frame(self.control_frame)
        self.file_label = ttk.Label(self.file_frame, text="Select Image:", style='Header.TLabel')
        self.file_button = ttk.Button(self.file_frame, text="Browse", command=self._browse_file, style='Primary.TButton')
        self.file_path_var = tk.StringVar(value="No file selected")
        self.file_path_label = ttk.Label(self.file_frame, textvariable=self.file_path_var, style='Info.TLabel')

        self.process_frame = ttk.Frame(self.control_frame)
        self.process_button = ttk.Button(self.process_frame, text="Colorize Image",
                                         command=self._process_image, style='Success.TButton', state='disabled')
        self.batch_button = ttk.Button(self.process_frame, text="Batch Process",
                                       command=self._batch_process, style='Warning.TButton')

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.control_frame, variable=self.progress_var, maximum=100)
        self.progress_label = ttk.Label(self.control_frame, text="", style='Info.TLabel')

        self.image_frame = ttk.LabelFrame(self.main_frame, text="Image Preview", padding="10")

        self.original_frame = ttk.Frame(self.image_frame)
        self.original_label = ttk.Label(self.original_frame, text="Original Image", style='Header.TLabel')
        self.original_canvas = tk.Canvas(self.original_frame, width=400, height=300, bg='white')

        self.colorized_frame = ttk.Frame(self.image_frame)
        self.colorized_label = ttk.Label(self.colorized_frame, text="Colorized Image", style='Header.TLabel')
        self.colorized_canvas = tk.Canvas(self.colorized_frame, width=400, height=300, bg='white')

        self.metrics_frame = ttk.LabelFrame(self.main_frame, text="Quality Metrics", padding="10")
        self.metrics_text = tk.Text(self.metrics_frame, height=6, width=50, font=('Courier', 9))

        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief='sunken')

    def _setup_layout(self):
        """Setup widget layout."""
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        self.title_label.pack(pady=(0, 20))
        self.control_frame.pack(fill='x', pady=(0, 20))
        self.file_frame.pack(fill='x', pady=(0, 10))
        self.file_label.pack(side='left', padx=(0, 10))
        self.file_button.pack(side='left', padx=(0, 10))
        self.file_path_label.pack(side='left')
        self.process_frame.pack(fill='x', pady=(0, 10))
        self.process_button.pack(side='left', padx=(0, 10))
        self.batch_button.pack(side='left')
        self.progress_bar.pack(fill='x', pady=(0, 5))
        self.progress_label.pack()
        self.image_frame.pack(fill='both', expand=True, pady=(0, 20))
        self.original_frame.pack(side='left', padx=(0, 10))
        self.original_label.pack()
        self.original_canvas.pack()
        self.colorized_frame.pack(side='right', padx=(10, 0))
        self.colorized_label.pack()
        self.colorized_canvas.pack()
        self.metrics_frame.pack(fill='x')
        self.metrics_text.pack()
        self.status_bar.pack(side='bottom', fill='x')

    def _browse_file(self):
        """Browse and select an image."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )

        if file_path:
            self.original_path = file_path
            self.file_path_var.set(os.path.basename(file_path))
            self._load_original_image()
            self.process_button.config(state='normal')
            self.status_var.set(f"Loaded: {os.path.basename(file_path)}")

    def _resize_image(self, image: Image.Image, max_size: Tuple[int, int] = (400, 300)) -> Image.Image:
        """Resize image while maintaining aspect ratio."""
        ratio = min(max_size[0] / image.size[0], max_size[1] / image.size[1])
        new_size = tuple(int(dim * ratio) for dim in image.size)
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def _update_canvas(self, canvas: tk.Canvas, image: Image.Image):
        """Update canvas with new image."""
        photo = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.create_image(
            canvas.winfo_width() // 2,
            canvas.winfo_height() // 2,
            image=photo,
            anchor="center"
        )
        canvas.image = photo  # Keep a reference to prevent garbage collection

    def _load_original_image(self):
        """Load and display the original image."""
        try:
            # Load image using PIL
            image = Image.open(self.original_path)
            self.original_image = image.copy()  # Store original for processing

            # Resize for display
            display_image = self._resize_image(image)
            self._update_canvas(self.original_canvas, display_image)

            # Clear colorized display and metrics
            self.colorized_canvas.delete("all")
            self.metrics_text.delete(1.0, tk.END)
            self.progress_var.set(0)
            self.progress_label.config(text="")
            
            logger.info(f"Successfully loaded image: {self.original_path}")
            self.status_var.set("Image loaded successfully")
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.status_var.set("Error loading image")

    def _cv2_to_pil(self, cv_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image to PIL Image."""
        if cv_image.shape[2] == 3:  # If BGR
            cv_image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
        return Image.fromarray(cv_image)

    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format."""
        # Convert PIL image to RGB numpy array
        rgb_image = np.array(pil_image.convert('RGB'))
        # Convert RGB to BGR
        bgr_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)
        return bgr_image

    def _update_metrics(self, original_cv: np.ndarray, colorized_cv: np.ndarray):
        """Calculate and display quality metrics."""
        try:
            # Calculate metrics using the colorizer's evaluation method
            metrics = self.colorizer.evaluate_colorization(original_cv, colorized_cv)

            # Clear previous metrics
            self.metrics_text.delete(1.0, tk.END)

            # Display metrics with formatting
            metrics_text = f"""Quality Metrics:

SSIM (Structural Similarity Index): {metrics['ssim']:.4f}
PSNR (Peak Signal-to-Noise Ratio): {metrics['psnr']:.2f} dB
Colorfulness Score: {metrics['colorfulness']:.2f}
"""
            self.metrics_text.insert(tk.END, metrics_text)

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, "Error calculating metrics")

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            self.metrics_text.insert(tk.END, f"Error calculating metrics: {str(e)}")

    def _display_color_histograms(self, original: np.ndarray, colorized: np.ndarray):
        """Display color histograms comparison."""
        # Create a new window for histograms
        histogram_window = tk.Toplevel(self.root)
        histogram_window.title("Color Histograms Comparison")
        histogram_window.geometry("800x400")

        # Create figure with subplots
        fig = Figure(figsize=(10, 4))
        
        # Plot original image histogram
        ax1 = fig.add_subplot(121)
        self._plot_color_histogram(original, ax1, "Original Image")
        
        # Plot colorized image histogram
        ax2 = fig.add_subplot(122)
        self._plot_color_histogram(colorized, ax2, "Colorized Image")

        # Add the plot to the window
        canvas = FigureCanvasTkAgg(fig, master=histogram_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_color_histogram(self, image: np.ndarray, ax, title: str):
        """Plot color histogram for a single image."""
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist, color=color, alpha=0.7)
        ax.set_xlim([0, 256])
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    def _process_image(self):
        """Process single image colorization."""
        if not self.original_image or not self.model_loaded:
            messagebox.showerror("Error", "Please load an image first and ensure the model is loaded.")
            return

        def process():
            try:
                # Update UI
                self.process_button.config(state='disabled')
                self.progress_var.set(0)
                self.progress_label.config(text="Processing...")
                self.status_var.set("Colorizing image...")

                # Convert PIL to OpenCV format
                cv_image = self._pil_to_cv2(self.original_image)

                # Process image
                colorized = self.colorizer.colorize_image(cv_image)
                
                # Convert back to PIL and display
                self.colorized_image = self._cv2_to_pil(colorized)
                display_image = self._resize_image(self.colorized_image)
                self._update_canvas(self.colorized_canvas, display_image)

                # Create grayscale version for metrics comparison
                original_gray = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)
                original_gray_3ch = cv.cvtColor(original_gray, cv.COLOR_GRAY2BGR)
                
                # Calculate and display quality metrics
                self._update_metrics(original_gray_3ch, colorized)

                # Update UI
                self.progress_var.set(100)
                self.progress_label.config(text="Complete!")
                self.status_var.set("Colorization complete")

                # Save the result
                output_path = os.path.join("output", "colorized_" + os.path.basename(self.original_path))
                os.makedirs("output", exist_ok=True)
                self.colorized_image.save(output_path)
                logger.info(f"Saved colorized image to: {output_path}")

            except Exception as e:
                logger.error(f"Error during colorization: {str(e)}")
                messagebox.showerror("Error", f"Failed to colorize image: {str(e)}")
                self.status_var.set("Error during colorization")
            finally:
                self.process_button.config(state='normal')

        # Run processing in a separate thread
        threading.Thread(target=process, daemon=True).start()

    def _batch_process(self):
        """Batch process multiple images."""
        if not self.model_loaded:
            messagebox.showerror("Error", "Please ensure the model is loaded first.")
            return

        # Select input directory
        input_dir = filedialog.askdirectory(title="Select Input Directory")
        if not input_dir:
            return

        # Select output directory
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return

        def process_batch():
            try:
                # Get list of image files
                image_files = [f for f in os.listdir(input_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                total_files = len(image_files)

                if total_files == 0:
                    messagebox.showinfo("Info", "No image files found in the selected directory.")
                    return

                # Process each image
                for i, filename in enumerate(image_files, 1):
                    try:
                        # Update progress
                        progress = (i - 1) / total_files * 100
                        self.progress_var.set(progress)
                        self.progress_label.config(text=f"Processing {i}/{total_files}: {filename}")
                        self.status_var.set(f"Processing batch: {i}/{total_files}")

                        # Load and process image
                        input_path = os.path.join(input_dir, filename)
                        image = cv.imread(input_path)
                        if image is None:
                            logger.warning(f"Failed to load image: {filename}")
                            continue

                        # Colorize
                        colorized = self.colorizer.colorize_image(image)

                        # Save result
                        output_path = os.path.join(output_dir, "colorized_" + filename)
                        cv.imwrite(output_path, colorized)
                        logger.info(f"Processed and saved: {output_path}")

                    except Exception as e:
                        logger.error(f"Error processing {filename}: {str(e)}")
                        continue

                # Complete
                self.progress_var.set(100)
                self.progress_label.config(text="Batch processing complete!")
                self.status_var.set("Batch processing complete")
                messagebox.showinfo("Success", f"Successfully processed {total_files} images")

            except Exception as e:
                logger.error(f"Error during batch processing: {str(e)}")
                messagebox.showerror("Error", f"Batch processing failed: {str(e)}")
                self.status_var.set("Error during batch processing")

        # Run batch processing in a separate thread
        threading.Thread(target=process_batch, daemon=True).start()


def main():
    """Launch the GUI application."""
    root = tk.Tk()
    app = ModernColorizationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
