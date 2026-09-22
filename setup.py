from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="evots-agent",
    version="1.0.0",
    description="EvoTS-Agent: A Self-Evolving LLM Agent for Financial Time Series Change Point Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="EvoTS-Agent Team",
    author_email="contact@evots-agent.dev",
    url="https://github.com/alikiany80/evots-agent",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "full": ["ruptures>=1.1.9", "pandas>=2.0.0", "matplotlib>=3.7.0"],
        "dev": ["pytest>=7.0", "pytest-cov", "black", "ruff", "mypy"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
