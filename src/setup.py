"""
setup.py
─────────
Allows the project to be installed as a package so that
`from src.xxx import` works correctly from any working directory.

Install in editable mode (recommended for development):
    pip install -e .

This is required when running on Modal or Lightning.ai where
the working directory may differ from the project root.
"""

from setuptools import setup, find_packages

setup(
    name="btc_ppo_smc_bot",
    version="1.0.0",
    description="BTCUSDT PPO Futures Trading Bot with SMC + GARCH + Kelly",
    packages=find_packages(exclude=["tests*", "data*", "logs*", "models*"]),
    python_requires=">=3.10",
    install_requires=[
        "gymnasium>=1.0.0",
        "stable-baselines3[extra]>=2.3.0",
        "python-binance>=1.0.19",
        "binance-futures-connector>=4.0.0",
        "arch>=7.0.0",
        "smartmoneyconcepts>=0.0.27",
        "ta>=0.10.2",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "ccxt>=4.3.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "tensorboard>=2.16.0",
        "pyarrow>=15.0.0",
        "streamlit>=1.32.0",
        "plotly>=5.20.0",
    ],
    extras_require={
        "modal": ["modal>=0.64.0"],
        "lightning": ["lightning-sdk>=0.0.7"],
        "dev": ["pytest>=8.0.0", "pytest-cov>=5.0.0"],
    },
)
