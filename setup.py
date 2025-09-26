from setuptools import setup, find_packages

setup(
    name="mt5_backtest",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'PySide6>=6.0.0',
        'pandas',
        'numpy',
        'matplotlib',
        'MetaTrader5',
        'tqdm',
        'pytest',
    ],
    entry_points={
        'console_scripts': [
            'mt5-backtest=mt5_backtest.gui.gui_app:main',
        ],
    },
    python_requires='>=3.8',
    author="Your Name",
    author_email="your.email@example.com",
    description="MT5 Multi-timeframe Backtesting Tool",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mt5-backtest",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
