from pydantic import BaseModel
from typing import Literal

class ModelWizard(BaseModel):
    _model_id: str
    loader: Literal['huggingface', 'unsloth']

    def __init__(self, model_id: str, loader: str):
        super().__init__(model_id=model_id, loader=loader)

    def load_model(self):
        if self.loader == 'huggingface':
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model = AutoModelForCausalLM.from_pretrained(self._model_id)
                tokenizer = AutoTokenizer.from_pretrained(self._model_id)
                return model, tokenizer
            except ImportError:
                raise ImportError("Please install the transformers library to use huggingface models")
        if self.loader == 'unsloth':
            try:
                from unsloth import FastLanguageModel
                model, tokenizer = FastLanguageModel.from_pretrained(self._model_id)
                return model
            except ImportError:
                raise ImportError("Please install the unsloth library to use unsloth models")

model = ModelWizard(model_id='blahblah', loader='huggingface')
