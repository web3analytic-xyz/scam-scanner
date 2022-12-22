import torch
from typing import Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer

from scamscanner.src.models import ScamScanner
from scamscanner.src.utils import bytecode_to_opcode, get_w3
from scamscanner import hub


class InferenceInput(BaseModel):
    contract: str = Field(
        ...,
        example='0x5e4e65926ba27467555eb562121fac00d24e9dd2',
        title='Address of a smart contract',
    )


class InferenceOutput(BaseModel):
    pred: int = Field(
        ...,
        example=False,
        title='Is the contract a scam?',
    )
    prob: float = Field(
        ...,
        example=0.5,
        title='Predicted probability for predicted label',
    )
    success: bool = Field(
        ...,
        example=True,
        title='Successfully performed inference',
    )


class ErrorResponse(BaseModel):
    r"""Error response for the API."""

    error: str = Field(
        ...,
        example=True,
        title='Error?',
    )
    message: str = Field(
        ...,
        example='',
        title='Error message',
    )
    traceback: Optional[str] = Field(
        None,
        example='',
        title='Detailed traceback of the error',
    )


app: FastAPI = FastAPI(
    title='ScamScanner',
    description='Predict whether a smart contract is a scam or not',
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
    bytecode = w3.eth.get_code(w3.toChecksumAddress(request.address))
    
    if bytecode.hex() == '0x':
        # Not a contract!
        return {'pred': -1, 'prob': -1, 'success': False}

    opcode = bytecode_to_opcode(bytecode=bytecode)

    # Encode the OP CODE
    feats = app.package['featurizer'].transform(opcode).toarray()
    feats = torch.from_numpy(feats).float().unsqueeze(1)  # shape: 1 x 286

    # Do inference
    logits = app.package['scamscanner'].forward(feats)
    pred_prob = torch.sigmoid(logits).item()  # number between 0 and 1
    pred = bool(round(pred_prob))

    return {'pred': pred, 'prob': pred_prob, 'success': True}

