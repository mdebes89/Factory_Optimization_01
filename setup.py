from setuptools import setup, find_packages

setup(
    name="factory_optimization_rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "gym>=0.21.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
        "pytest>=7.0.0"
    ],
    python_requires=">=3.8",
)
