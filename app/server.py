from typing import Dict, Any, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from transformers import LongformerModel, LongformerTokenizer
from scamscanner.src.models import ScamScanner


class InferenceInput(BaseModel):
    contract_address: str = Field(
        ...,
        example='0x5e4e65926ba27467555eb562121fac00d24e9dd2',
        title='Address of a smart contract',
    )


class InferenceResult(BaseModel):
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


class InferenceResponse(BaseModel):
    error: str = Field(
        ...,
        example=False,
        title='Error?',
    )
    results: Dict[str, Any] = Field(
        ...,
        example={}, 
        title='Predicted label and probability results',
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

    longformer = LongformerModel.from_pretrained("allenai/longformer-base-4096")
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    scamscanner = ScamScanner.load_from_checkpoint('TODO')
    scamscanner.eval()

    app.package = {
        'longformer': longformer,
        'tokenizer': tokenizer,
        'scamscanner': scamscanner,
    }


@app.post(
    '/api/predict',
    response_model = InferenceResponse,
    responses = {
        422: {'model': ErrorResponse},
        500: {'model': ErrorResponse}
    }
)
def predict(request: Request, body: InferenceInput):
    pass
