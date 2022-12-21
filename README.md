# ScamScanner

Trains a classifier on contract OPCODES to classify if a smart contract is a phishing or malicious scam.

## Data

The raw dataset of Ethereum scam contracts are downloaded from various [forta-network](https://github.com/forta-network/labelled-datasets). A dataset of positive (non-scam) Ethereum contracts are downloaded from [tintinweb](https://github.com/tintinweb/smart-contract-sanctuary-ethereum). 

We preprocess these datasets extensively to fetch ABIs, OPCODES, get pretrained embeddings, and more. We store these larger files using `git-lfs`. When cloning the repo, please run `git lfs pull`.

## Usage

To train the ScamScanner model, run the following:
```
python scripts/train.py ./scamscanner/configs/train.yml --devices 0
```
We strongly recommend using a GPU as this can be prohibitively slow without one.

To evaluate a trained ScamScanner model, run the following:
```
python scripts/eval.py <checkpoint-file> --devices 0
```
We include a trained checkpoint in `./scamscanner/trained/checkpoint.pth` that can be used. 20\% of the dataset is randomly set aside as the test set, which the trained model did not get to see. 

To do live inference, we setup a simple FastAPI that loads the model and any necessary dependencies. To run the server, initialize the server:
```
uvicorn app/server:app --reload
```
You can then send API requests to your server e.g.
```
curl -X 'POST' \
  'http://127.0.0.1:8000/api/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "contract": "0x5e4e65926ba27467555eb562121fac00d24e9dd2"
  }'
```

## About

ScamScanner embeds contract OPCODES using a pretrained [LongFormer](https://arxiv.org/abs/2004.05150) from [Huggingface](https://huggingface.co/docs/transformers/model_doc/longformer). The ScamScanner model consists of stacked [Conformer](https://arxiv.org/abs/2105.03889) encoder layers followed by a [pooling layer](https://github.com/huggingface/transformers/blob/31d452c68b34c2567b62924ee0df40a83cbc52d5/src/transformers/models/longformer/modeling_longformer.py#L1372), and a final linear map to predict 0 (not a scam) or 1 (scam). 

Limited model and hypeparameter search was conducted. Further experiments should leverage better performance still. We emphasize that this code is a proof-of-concept, and should not be used at scale. Increasing the size of the labeled dataset would likely yield a more powerful classifier.
