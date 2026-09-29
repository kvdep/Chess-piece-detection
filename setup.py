from setuptools import find_packages, setup

setup(
    name="chess_piece_detection",
    version="1.0.0",
    description="Chess piece detection and board localization using YOLOv11, SSD, and Faster R-CNN",
    author="Alexander Kydysiuk",
    author_email="numl2202@gmail.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.3.0",
        "opencv-python>=4.8.0",
        "Pillow>=9.5.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
)
