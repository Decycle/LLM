This is a collection of 3 experiments that all aim to improve the model's ability to follow instructions better. The experiments are as follows:

- `DPO`: a fine-tuning technique introduces in the paper [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290). It allows fine-tuning model to achieve the same result as reinforcement learning without the need to train a separate reward model.

- `Control Vectors`: an implementation of the paper [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405). The paper is about extracting a "control vector" that can be used to influence the model's behavior in a predictable way (making it more honest, etc.)

- `Anticipation`: an experiment to see if it's possible to make the models "think ahead" by exploring the similarities between the hidden states at different token positions. It is ultimately abandoned because I don't think it aligns with how human language works.