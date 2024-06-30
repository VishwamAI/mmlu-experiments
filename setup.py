from setuptools import setup, find_packages

setup(
    name="mmlu",
    version="0.1.0",
    description="A library for MMLU functionalities",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "pytest",
        "flake8"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/VishwamAI/mmlu-experiments",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
