# %%
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
import wandb
from torch.nn import CosineSimilarity
from scipy.stats import spearmanr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "princeton-nlp/unsup-simcse-roberta-large"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)

# %%
BATCH_SIZE = 16
EPCOH = 1
LR = 1e-5
TEMPERATURE = 0.05
MAX_LEN = 512
VALIDATION_SIZE = 40


dataset = load_dataset("wikipedia", "20220301.en", split="train[:1000000]")
dataset = dataset.shuffle()


TOKENIZER_BATCH_MULTIPLIER = 16

token_batch_size = BATCH_SIZE * TOKENIZER_BATCH_MULTIPLIER
token_num_batches = len(dataset) // token_batch_size

def get_batch(dataset):

    input_ids = torch.zeros((token_num_batches, token_batch_size, MAX_LEN), dtype=torch.long)
    attention_masks = torch.zeros((token_num_batches, token_batch_size, MAX_LEN), dtype=torch.long)

    for i in tqdm(range(0, len(dataset), token_batch_size)):
        if i // token_batch_size >= token_num_batches:
            break
        data = dataset[i:i+token_batch_size]
        data = data['text']
        data = tokenizer(data, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
        input_ids[i//token_batch_size] = data['input_ids']
        attention_masks[i//token_batch_size] = data['attention_mask']

    print(input_ids.shape)
    input_ids = input_ids.reshape((-1, BATCH_SIZE, MAX_LEN))
    attention_masks = attention_masks.reshape((-1, BATCH_SIZE, MAX_LEN))
    return input_ids, attention_masks

output_string = """
###################
# TOKENIZING DATA #
###################
"""

print(output_string.strip())

input_ids, attention_masks = get_batch(dataset)

input_ids = input_ids.to(device)
attention_masks = attention_masks.to(device)

num_batches = input_ids.shape[0]

torch.save(input_ids, '/workspace/data/all_512/input_ids.pt')
torch.save(attention_masks, '/workspace/data/all_512/attention_masks.pt')

# input_ids = torch.load('/workspace/data/all_512/input_ids.pt')
# attention_masks = torch.load('/workspace/data/all_512/attention_masks.pt')

wandb.init(project="simcse")


def log_graph(model, step):
    input_string = """
    The reason for the high price of oil is the war in the Middle East. As the war continues, the price of oil will continue to rise. Authorities say that the price of oil will reach $100 per barrel. The price of oil has already reached $80 per barrel.
    """

    second_string = """
    do you know why I like you? Because you are this shining star in my life. You are the one who makes me happy. You are the one who makes me feel good. You are the one who makes me feel special. You are the one who makes me fee
    """

    input_ids = tokenizer(input_string, return_tensors="pt").to(device).input_ids
    input_ids2 = tokenizer(second_string, return_tensors="pt").to(device).input_ids
    with torch.no_grad():
        output1 = model(input_ids).last_hidden_state[0].detach().cpu().numpy()
        output2 = model(input_ids).last_hidden_state[0].detach().cpu().numpy()
        output3 = model(input_ids2).last_hidden_state[0].detach().cpu().numpy()

    outputs = np.concatenate([output1, output2, output3])
    labels = np.concatenate([
        np.zeros(output1.shape[0]),
        np.ones(output2.shape[0]) * 0.5,
        np.ones(output3.shape[0]),
    ])

    pca = PCA(n_components=2)

    outputs_2d = pca.fit_transform(outputs)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].scatter(outputs_2d[:, 0], outputs_2d[:, 1], c=labels)

    output_2d = pca.fit_transform(output1)
    axes[1].scatter(output_2d[:, 0], output_2d[:, 1], c=np.arange(output_2d.shape[0]))

    wandb.log({
        'graph': wandb.Image(fig),
    })

    # close
    plt.close(fig)

dataset = load_dataset("stsb_multi_mt", 'en', split="test").with_format(
    "torch", device=device)

def prepare_validation_set(dataset, max_size=VALIDATION_SIZE):
    batch = []
    for i in range(max_size * BATCH_SIZE):
        data = dataset[i]
        premise = tokenizer(data['sentence1'], padding='max_length', truncation=True,
                            return_tensors='pt', max_length=MAX_LEN)
        hypothesis = tokenizer(data['sentence2'], padding='max_length', truncation=True,
                               return_tensors='pt', max_length=MAX_LEN)
        batch.append((premise, hypothesis, data['similarity_score']))

        if len(batch) == BATCH_SIZE:
            token_ids = torch.zeros(2, BATCH_SIZE, MAX_LEN, dtype=torch.long)
            token_masks = torch.zeros(2, BATCH_SIZE, MAX_LEN, dtype=torch.long)
            scores = torch.zeros(BATCH_SIZE, dtype=torch.float)

            for i, (premise, hypothesis, score) in enumerate(batch):
                token_ids[0, i] = premise['input_ids']
                token_ids[1, i] = hypothesis['input_ids']

                token_masks[0, i] = premise['attention_mask']
                token_masks[1, i] = hypothesis['attention_mask']

                scores[i] = score

            yield token_ids, token_masks, scores
            batch = []

cos = CosineSimilarity()
validation_set = list(prepare_validation_set(dataset))

def validate(model, step=0, log=True):
    model.eval()
    with torch.no_grad():
        predicated_empty_scores = []
        predicated_last_scores = []
        predicated_mutual_last_scores = []
        actual_scores = []
        for batch in validation_set:
            token_ids, token_masks, scores = batch
            token_ids = token_ids.to(device)
            token_masks = token_masks.to(device)
            scores = scores.to(device)

            sentence1_outputs = model(
                input_ids=token_ids[0], attention_mask=token_masks[0]).last_hidden_state

            sentence2_outputs = model(
                input_ids=token_ids[1], attention_mask=token_masks[1]).last_hidden_state

            empty_sentence1_output = sentence1_outputs[:, -1]
            empty_sentence2_output = sentence2_outputs[:, -1]

            predicated_empty_score = cos(empty_sentence1_output, empty_sentence2_output)
            predicated_empty_scores.extend(predicated_empty_score.cpu().numpy())

            length1 = token_masks[0].sum(dim=1)
            length2 = token_masks[1].sum(dim=1)

            last_sentence1_output = sentence1_outputs[torch.arange(sentence1_outputs.shape[0]), length1 - 1]
            last_sentence2_output = sentence2_outputs[torch.arange(sentence2_outputs.shape[0]), length2 - 1]

            predicated_last_score = cos(last_sentence1_output, last_sentence2_output)
            predicated_last_scores.extend(predicated_last_score.cpu().numpy())

            shortest_length = torch.min(length1, length2)

            mutual_last_sentence1_output = sentence1_outputs[torch.arange(sentence1_outputs.shape[0]), shortest_length - 1]
            mutual_last_sentence2_output = sentence2_outputs[torch.arange(sentence2_outputs.shape[0]), shortest_length - 1]

            predicated_mutual_last_score = cos(mutual_last_sentence1_output, mutual_last_sentence2_output)
            predicated_mutual_last_scores.extend(predicated_mutual_last_score.cpu().numpy())

            actual_scores.extend(scores.cpu().numpy())


        rank_empty = spearmanr(predicated_empty_scores, actual_scores)
        rank_last = spearmanr(predicated_last_scores, actual_scores)
        rank_mutual_last = spearmanr(predicated_mutual_last_scores, actual_scores)
    model.train()
    if log:
        wandb.log({
            'rank_empty': rank_empty.correlation,
            'rank_last': rank_last.correlation,
            'rank_mutual_last': rank_mutual_last.correlation,
        })
    return rank_empty.correlation, rank_last.correlation, rank_mutual_last.correlation
# %%
from torch import cuda
import gc

gc.collect()
cuda.empty_cache()
model.zero_grad(set_to_none=True)

from torch.optim import AdamW
model.train()

optimizer = AdamW(model.parameters(), lr=LR)

output_string = """
##################
# START TRAINING #
##################
"""

print(output_string.strip())

for epoch in range(EPCOH):
    for i in tqdm(range(num_batches)):

        optimizer.zero_grad()

        input_id = input_ids[i]
        attention_mask = attention_masks[i]

        embeddings_1 = model(input_id, attention_mask).last_hidden_state
        embeddings_2 = model(input_id, attention_mask).last_hidden_state

        embeddings_1 = embeddings_1 / torch.norm(embeddings_1, dim=-1, keepdim=True)
        embeddings_2 = embeddings_2 / torch.norm(embeddings_2, dim=-1, keepdim=True)

        embeddings_1 = embeddings_1 * attention_mask.unsqueeze(-1)
        embeddings_2 = embeddings_2 * attention_mask.unsqueeze(-1)

        embeddings_1 = embeddings_1.reshape(-1, 1024)
        embeddings_2 = embeddings_2.reshape(-1, 1024)

        similarity = torch.mm(embeddings_1, embeddings_2.T)
        similarity = similarity.reshape(BATCH_SIZE, MAX_LEN, BATCH_SIZE, MAX_LEN)

        similarity = torch.exp(similarity / 0.05)

        similarity = torch.sum(similarity, dim=(1, 3)) / torch.sum(attention_mask, dim=-1)

        chosen_sim = torch.diagonal(similarity)
        sum_sim = torch.sum(similarity, dim=-1)

        loss = -torch.log(chosen_sim / sum_sim)
        loss = torch.mean(loss)

        loss.backward()
        optimizer.step()

        wandb.log({
            'loss': loss.item(),
        })
        if i % 500 == 0:
            print(f"Validation, iter: {i}")
            validate(model, i + 1)
        if i % 1000 == 0:
            print(f"Graph, iter: {i}")
            log_graph(model, i + 1)
        if i % 4000 == 3999:
            print(f"Save Model, iter: {i}")
            model.save_pretrained(f'/workspace/results/{i}')


# %%
model.save_pretrained('/workspace/results/last')


