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
    model_path = hub.get_model('pretrained-12-22')
    scamscanner = ScamScanner.load_from_checkpoint(model_path)
    scamscanner.eval()

    app.package = {'scamscanner': scamscanner}


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
        return {'pred': -1, 'prob': -1}

    opcode = bytecode_to_opcode(bytecode=bytecode)

    # Encode the OP CODE
    encoded = app.package['tokenizer'](opcode)

    # Create a batch of 1
    input_ids = encoded['input_ids'].unsqueeze(0)
    attention_mask = encoded['attention_mask'].unsqueeze(0)

    # Do inference
    outputs = app.package['longformer'](input_ids, attention_mask=attention_mask)
    sequence_output = outputs.last_hidden_state
    pad_mask = torch.ones(1, sequence_output.size(1)).long()

    batch = {'emb': sequence_output, 'emb_mask': pad_mask}
    logit = app.package['scamscanner'].forward(batch)
    pred_prob = torch.sigmoid(logit).item()  # number between 0 and 1
    pred = bool(round(pred_prob))

    # Not a contract!
    return {'pred': pred, 'prob': pred_prob}
