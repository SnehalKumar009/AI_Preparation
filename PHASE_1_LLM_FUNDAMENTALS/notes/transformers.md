# Transformers & Attention (sample note)

The Transformer architecture, introduced in the 2017 paper "Attention Is All
You Need", replaced recurrence with self-attention. Self-attention lets every
token attend to every other token in the sequence, computing a weighted sum of
value vectors where the weights come from the similarity between query and key
vectors.

## Key ideas

- **Tokens**: text is split into subword tokens by a tokenizer (e.g. BPE).
- **Embeddings**: each token id maps to a dense vector. Positional information
  is added so the model knows token order.
- **Context window**: the maximum number of tokens the model can attend to at
  once. Prompts longer than this must be truncated or summarized.
- **Attention heads**: multiple attention operations run in parallel, each
  learning different relationships.

## Decoding controls

- **Temperature** scales the logits before softmax. Higher temperature spreads
  probability mass, producing more varied output. Temperature 0 is greedy.
- **Top-p (nucleus) sampling** keeps the smallest set of tokens whose
  cumulative probability exceeds p, then samples among them.

## Why RAG

LLMs hallucinate when asked about facts outside their training data. Retrieval
Augmented Generation grounds answers by retrieving relevant text and injecting
it into the prompt, so the model quotes real sources instead of inventing them.
