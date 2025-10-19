"""
Minimal setup.py for emailops package.

This allows running 'pip install -e .' for development installation.
"""

from setuptools import find_packages, setup

setup(
    name="emailops",
    version="0.1.0",
    packages=find_packages(include=["emailops", "emailops.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.20.0",
        "google-cloud-aiplatform>=1.35.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "ruff>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "emailops-gui=emailops.emailops_gui:main",
        ],
    },
)
