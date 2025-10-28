"""
Setup script for Advanced Image Colorization System
Academic Deep Learning Project
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="advanced-image-colorization",
    version="1.0.0",
    author="[Your Name]",
    author_email="[your.email@college.edu]",
    description="Advanced deep learning-based image colorization system",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/[your-username]/advanced-image-colorization",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "colorize=src.colorizer:main",
            "colorize-gui=src.gui:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.prototxt", "*.caffemodel", "*.npy"],
    },
    keywords="deep-learning, computer-vision, image-processing, colorization, neural-networks",
    project_urls={
        "Bug Reports": "https://github.com/[your-username]/advanced-image-colorization/issues",
        "Source": "https://github.com/[your-username]/advanced-image-colorization",
        "Documentation": "https://github.com/[your-username]/advanced-image-colorization#readme",
    },
) 