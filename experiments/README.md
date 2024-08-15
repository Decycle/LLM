This is a folder contains the more random experiments that isn't studied enough to have their own folder. Right now it contains:

- `Context Aware Embedding`: Embeddings techniques that are aware of a certain "context sentence" or "seed sentence". If it's successful we can use this to embed long articles by splitting them into smaller parts and embeddings the later parts with beginning parts as context.
- `Long Article Embedding`: a continuation of the previous experiment.
- `Remove Layer Experiment`: Remove certain layers of the LLM model by settings the weights to zero and observe its capabilities with the MMLU dataset. Surprisingly, the model still performs well with a lot of layers removed.
- `First Person`: Experimenting on making the model generates text in first person. Result is not satisfactory.
