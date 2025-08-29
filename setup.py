from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dcmbench",
    version="0.1.2",
    packages=find_packages(
        include=["dcmbench", "dcmbench.*"],
        exclude=["tests", "tests.*", "examples", "examples.*", "Aug25_test", "Aug25_test.*"]
    ),
    author="Carlos Guirado",
    author_email="your.email@example.com",
    description="A comprehensive benchmarking framework for discrete choice models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carlosguirado/dcmbench",
    project_urls={
        "Bug Tracker": "https://github.com/carlosguirado/dcmbench/issues",
        "Documentation": "https://dcmbench.readthedocs.io",
        "Source": "https://github.com/carlosguirado/dcmbench",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "biogeme>=3.2.0",
        "requests>=2.25.0",
        "scipy>=1.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
        "pytorch": [
            "torch>=1.10.0",
        ],
    },
    include_package_data=True,
    package_data={
        "dcmbench": ["datasets/metadata.json"],
    },
    keywords="discrete choice benchmarking transportation econometrics machine learning",
)