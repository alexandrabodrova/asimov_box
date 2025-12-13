"""
LegalBreak: Law-Aware Adversarial Testing for LLM Legal Compliance

A systematic framework for measuring LLM vulnerabilities to legal compliance
violations across dual-use content, copyright infringement, and defamation.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="legalbreak",
    version="0.1.0",
    author="Alexandra Bodrova",
    author_email="",
    description="Law-aware adversarial testing framework for LLM legal compliance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexandrabodrova/asimov_box",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "viz": [
            "pandas>=2.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "legalbreak=legalbreak.core.legal_guarddog_core:main",
        ],
    },
    include_package_data=True,
    keywords="llm adversarial-testing legal-compliance ai-safety jailbreaking",
)
