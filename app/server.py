import torch
import joblib
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from scamscanner.src.models import ScamScanner
from scamscanner.src.utils import bytecode_to_opcode, get_w3
from scamscanner import hub


class InferenceInput(BaseModel):
    contract: str = Field(
        ...,
        example='0x5e4e65926ba27467555eb562121fac00d24e9dd2',
        title='Address of the smart contract to make a prediction for.',
    )


class InferenceOutput(BaseModel):
    pred: int = Field(
        ...,
        example=0,
        title='Takes value 1 if contract is malicious, 0 if not, and -1 if address is not a contract.',
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


class ErrorResponse(BaseModel):
    r"""Error response for the API."""

    error: bool = Field(
        ...,
        example=True,
        title='Is there an error?',
    )
    message: str = Field(
        ...,
        example='',
        title='Error message.',
    )
    traceback: Optional[str] = Field(
        None,
        example='',
        title='Detailed traceback of the error.',
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


@app.post(
    '/api/predict',
    response_model = InferenceOutput,
    responses = {
        422: {'model': ErrorResponse},
        500: {'model': ErrorResponse}
    })
def predict(request: Request, body: InferenceInput):
    w3 = get_w3()

    # Get the OPCODE for the contract
    bytecode = w3.eth.get_code(w3.toChecksumAddress(body.contract))
    
    if bytecode.hex() == '0x':
        # Not a contract!
        return {'pred': -1, 'prob': -1, 'success': False}

    opcode = bytecode_to_opcode(bytecode=bytecode)

    # Encode the OP CODE
    feats = app.package['featurizer'].transform([opcode]).toarray()
    feats = torch.from_numpy(feats).float()  # shape: 1 x 286

    # Do inference
    logits = app.package['scamscanner'].forward({'feat': feats})
    pred_prob = torch.sigmoid(logits).item()  # number between 0 and 1
    pred = bool(round(pred_prob))

    return {'pred': pred, 'prob': pred_prob, 'success': True}

