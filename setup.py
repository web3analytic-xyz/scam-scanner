from setuptools import setup, find_packages

LONG_DESCRIPTION = open("README.md", "r").read()

setup(
    name="scamscanner",
    version="0.1.0",
    author="grubiroth",
    author_email="mike@paretolabs.xyz",
    packages=find_packages(),
    scripts=[],
    description="Uses embedding models on OPCODES to classify if a smart contract is a phishing or malicious scam.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/web3analytic-xyz/scam-contract-classifier",
    install_requires=[
    ]
)
