"""Setup script for Newspaper Tension package."""

from setuptools import setup, find_packages
import os

with open("README.MD", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="Newspaper_Tension",
    version="1.0.0",
    author="Peter",
    description="BERT sentiment analysis of GDELT News for country tension prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.0.0",
        "pandas",
        "numpy",
        "matplotlib",
        "tqdm",
        "torchvision",
        "torchtext",
        "newspaper3k>=0.2.8",
        "requests",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    }
)
