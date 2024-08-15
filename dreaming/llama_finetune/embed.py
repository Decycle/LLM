import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModel, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "princeton-nlp/sup-simcse-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.to(device)
model.eval()


def embed(sentence):
    with torch.no_grad():
        input = tokenizer(sentence, return_tensors="pt", padding=True)
        input_ids = input["input_ids"].to(device)

        outputs = model(input_ids)
        embeddings = outputs.pooler_output
        embeddings = F.normalize(embeddings)
        return embeddings.cpu().numpy()[0]


print(embed("I like to eat apples.").shape)
