"""Setup file for GMMD package
"""
# Import modules
# --------------
from setuptools import setup, find_packages
import os

# Get path of the package, where steup.py is located
here = os.path.abspath(os.path.dirname(__file__))
# Read the verison number
with open(os.path.join(here, "VERSION")) as versionFile:
    version = versionFile.read().strip()
# Store the README.md file
with open(os.path.join(here, "readme.md"), encoding="utf-8") as f:
    longDescription = f.read()
setup(
    # Project name
    name="geommicgen",
    # Version from the version file
    version=version,
    # Short description
    description="GMMD: Generation of Microstructures using Molecular Dynamics",
    # Long descriptionf from README.md
    long_description=longDescription,
    long_description_content_type="text/markdown",
    # Github url
    url="https://github.com/josevilacha/GMMD",
    # Authors
    author="José Luís P. Vila-Chã, Bernardo P. Ferreira, Francisco M. Andrade Pires @CM2S, FEUP",
    author_email="jvc@fe.up.pt, bpferreira@fe.up.pt, fpires@fe.up.pt",
    # Licensing
    licence="BSD 3-Clause License",
    # Classifiers (selected from https://pypi.org/classifiers/)
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        # Python version obtained with https://pypi.org/project/check-python-versions/
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Microstructure Generation",
    ],
    # Keywords
    keywords="Microstructure generation",
    # Project URLs
    project_urls={
        # 'Documentation': 'https://packaging.python.org/tutorials/distributing-packages/',
        "Source": "https://github.com/josevilacha/GMMD",
        "Tracker": "unknown",
    },
    # Python version compatibility
    python_requires=">=3.8, <3.9",
    # Source directory
    # package_dir={"": "src"},
    # Packages provided
    packages=find_packages(where="src"),
    # Execution command
    entry_points={
        "console_scripts": [
            "geommicgen=geommicgen:run",
        ],
    },
    install_requires=[
        "cycler>=0.10.0",
        "kiwisolver>=1.3.1",
        "matplotlib>=3.4.2",
        "numpy>=1.21.0",
        "Pillow>=8.2.0",
        "pyparsing>=2.4.7",
        "python-dateutil>=2.8.1",
        "scipy>=1.7.0",
        "six>=1.16.0",
        "tabulate>=0.8.9",
    ],
)
