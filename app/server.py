import torch
import joblib

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from scamscanner.src.models import ScamScanner
from scamscanner.src.utils import bytecode_to_opcode, get_w3
from scamscanner import hub


class InferenceOutput(BaseModel):
    pred: int = Field(
        ...,
        example=0,
        title='Takes value 1 if contract is malicious, 0 if not.',
    )
    prob: float = Field(
        ...,
        example=0.5,
        title='Predicted probability for the contract being malicious.',
    )
    success: bool = Field(
        ...,
        example=True,
        title='Successfully performed inference',
    )


app: FastAPI = FastAPI(
    title='ScamScanner',
    description='Predict whether a smart contract is a scam or not',
)

# For interacting with the frontend
origins = [
    "http://localhost:3000",
    "localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.on_event("startup")
async def startup_event():
    # Load trained scam-scanner model
    model_path = hub.get_model('pretrained-12-22')
    scamscanner = ScamScanner.load_from_checkpoint(model_path)
    scamscanner.eval()

    # Load featurizer
    featurizer = joblib.load(hub.get_model('featurizer-12-22'))

    app.package = {'scamscanner': scamscanner, 'featurizer': featurizer}


@app.get('/api/scan/{contract}', response_model = InferenceOutput)
def scan(contract: str, request: Request):
    r"""Call trained ScamScanner model to predict if a contract is malicious."""

    if not verify_contract(contract):
        raise HTTPException(status_code=400, detail='Invalid contract address')

    w3 = get_w3()

    # Get the OPCODE for the contract
    bytecode = w3.eth.get_code(w3.toChecksumAddress(contract))

    if bytecode.hex() == '0x':
        # Not a contract!
        raise HTTPException(status_code=400, detail='Address is not a contract')

    opcode = bytecode_to_opcode(bytecode=bytecode)

    # Encode the OP CODE
    feats = app.package['featurizer'].transform([opcode]).toarray()
    feats = torch.from_numpy(feats).float()  # shape: 1 x 286

    # Do inference
    logits = app.package['scamscanner'].forward({'feat': feats})
    pred_prob = torch.sigmoid(logits).item()  # number between 0 and 1
    pred = round(pred_prob)

    return {'pred': pred, 'prob': pred_prob, 'success': True}


def verify_contract(contract):
    r"""Simple validation on contract address input.
    Returns:
    --
    is_valid (bool)
    """
    if not isinstance(contract, str):
        return False

    if len(contract) != 42:
        return False

    if contract[:2] != '0x':
        return False

    return True
