# ScamScanner

Trains a classifier on contract OPCODES to classify if a smart contract is a phishing or malicious scam.

## Data

The raw dataset of Ethereum scam contracts are downloaded from various [forta-network](https://github.com/forta-network/labelled-datasets). A dataset of positive (non-scam) Ethereum contracts are downloaded from [tintinweb](https://github.com/tintinweb/smart-contract-sanctuary-ethereum). 

We preprocess these datasets extensively to fetch ABIs, bytecodes, and OPCODES. We store these larger files using `git-lfs`. When cloning the repo, please run `git lfs pull`.

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
We include a trained checkpoint in `./scamscanner/hub` that can be used. 20\% of the dataset is randomly set aside as the test set, which the trained model did not get to see. This command will output the loss and accuracy on the test set.

To do live inference, we setup a simple FastAPI that loads the model and any necessary dependencies. To run the server, initialize the server:
```
uvicorn app.server:app --reload
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

## Performance

The provided checkpoint (trained for 200 epochs) achieves the following:
```
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
        test/acc            0.9163498098859315
         test/f1            0.9150579150579151
        test/loss           0.3454396426677704
     test/precision         0.9294117647058824
       test/recall          0.9011406844106464
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

## About

ScamScanner embeds contract OPCODES using a residual MLP network on top of TF-IDF features computed from a training set of smart contract OPCODES. We opt for this simple model (as opposed to a sequential or attention model) due to the limited size of our dataset (~1k positive examples). Ablation experiemnts found TF-IDF to outperform bag-of-words features due to normalization challenges.

Limited model and hyperparameter search were conducted. Further experiments should leverage better performance still. We emphasize that this code is a proof-of-concept, and should not be used at scale. Increasing the size of the labeled dataset and utilizing a transformer-based model is an interesting direction for future work.
