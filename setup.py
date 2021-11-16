from setuptools import setup, find_packages

setup(
    name="cellseg",
    version="0.1.0",
    packages=["cellseg"],
    install_requires=[
        "numpy",
        "pandas",
        "omegaconf",
        "scikit-learn",
        "matplotlib",
        "hydra-core",
        "efficientnet-pytorch",
        "torch@https://download.pytorch.org/whl/cu113/torch-1.10.0%2Bcu113-cp39-cp39-linux_x86_64.whl",
        "torchvision@https://download.pytorch.org/whl/cu113/torchvision-0.11.1%2Bcu113-cp39-cp39-linux_x86_64.whl",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-mypy",
            "black",
            "pytest-cov",
            "pytest-benchmark",
            "diagrams",
            "mypy",
            "kaggle",
        ]
    },
)
