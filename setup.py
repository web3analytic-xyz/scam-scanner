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
        'dotmap==1.3.30',
        'evmdasm==0.1.10',
        'fastapi==0.88.0',
        'joblib==1.2.0',
        'jsonlines==3.1.0',
        'numpy==1.23.4',
        'pandas==1.5.2',
        'pydantic==1.10.2',
        'pytorch_lightning==1.8.6',
        'PyYAML==6.0',
        'requests==2.28.1',
        'scikit_learn==1.2.0',
        'setuptools==65.6.3',
        'torch==1.12.1',
        'tqdm==4.64.1',
        'web3==5.31.3',
        'uvicorn==0.20.0',
    ]
)
