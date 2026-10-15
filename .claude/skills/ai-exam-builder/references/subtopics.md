# Domains, subtopics and block positions

Topic numbers below are the ones the user uses: "Topic 6" means Deep Learning Advanced.

In combined mode, domains appear in **blocks in this order**, not interleaved. Roughly 10–15% of questions get displaced into a neighbouring block so the boundary is not exactly where the count predicts — but the blocked trend stays obvious.

| Topic | Domain | Block | Count |
|---|---|---|---|
| 1 | Statistical Thinking and Applications | Q1–Q9 | 9 |
| 2 | Beyond LLMs: Prompts, Agents, RAG | Q10–Q17 | 8 |
| 3 | RAI: Governance and Ethical Fundamentals | Q18–Q25 | 8 |
| 4 | PyTorch / TensorFlow Fundamentals | Q26–Q33 | 8 |
| 5 | Machine Learning Fundamentals | Q34–Q41 | 8 |
| 6 | Deep Learning Advanced | Q42–Q50 | 9 |
| 7 | RAI: Technical Reliability and Explainability | Q51–Q58 | 8 |
| 8 | Natural Language Processing | Q59–Q66 | 8 |
| 9 | Computer Vision Fundamentals | Q67–Q74 | 8 |
| 10 | Large Language Models (Tech) | Q75–Q82 | 8 |
| 11 | Retrieval-Augmented Generation | Q83–Q90 | 8 |

Domains are **unlabelled** in the question section. Tags appear only in the answer key.

## Code expectation per topic

Not every domain suits code. Do not force a code block into a topic where it is unnatural; carry reading load there through data tables, realistic scenarios and output snippets instead.

| Topics | Code questions per block |
|---|---|
| 1, 3, 7 | 0. Reading load comes from data tables, scenarios and output snippets. |
| 2, 11 | About 2, only where natural (a tool-call loop, a chunking or retrieval snippet). Prompt-ranking sets stay prompt text. |
| 4, 5, 8, 9, 10 | 5 of 8 |
| 6 | 6 of 9 |

This sums to 35 code questions across the 90-question paper.

---

## Coverage surface

A question's primary subject must sit inside its topic's subtopics. Other topics may be drawn on **only** as supporting material for a multi-concept question — a Deep Learning question that also needs a tensor-shape trace, a Statistics question framed around a churn model. The question still belongs to the requested topic.

Cover the topic's full width across a set. For more than ten questions, structure in batches of ten, each batch independently spanning the topic rather than running easy-first then hard. A requested count is a target, not a quota to fill with weak items.

### Topic 1 — Statistical Thinking and Applications Fundamentals
Descriptive statistics, probability, Bayes' theorem, normal/binomial/Poisson distributions, matrix operations, multiplication and transpose, hypothesis testing, p-values, correlation, confidence intervals, ANOVA, matrix decomposition, linear and logistic regression, sampling techniques, A/B testing.

### Topic 2 — Beyond LLMs: Prompts, Agents, and RAG
Prompt engineering for different use cases, RAG fundamentals, chunking strategies, vector databases, agents, basic tool use, retrieval vs. generation, end-to-end RAG flow, embeddings, similarity search, RAG evaluation, hybrid search, reranking.

This topic carries the prompt-ranking questions. Read `prompt-ranking.md`.

### Topic 3 — RAI: Governance and Ethical Fundamentals
AI governance, ethical considerations, responsible use of AI.

### Topic 4 — PyTorch Fundamentals or TensorFlow Fundamentals
Tensors, tensor operations, simple neural networks, `nn.Module`, datasets and dataloaders, training loops, custom datasets, saving/loading models, basic CNN implementation, loss functions, optimizers, GPU basics, debugging training issues, simple transfer learning.

### Topic 5 — Machine Learning Fundamentals
Supervised vs. unsupervised learning, train/test split, linear and logistic regression, decision trees, k-NN, k-means, evaluation metrics, SVM, random forests, cross-validation, boosting, bagging, bias-variance trade-off.

### Topic 6 — Deep Learning Advanced
ANN basics, perceptron, activation and loss functions, forward pass, backpropagation, gradient descent, CNNs, kernels/filters, padding, strides, pooling, basic CNN architecture, RNNs, LSTM/GRU, overfitting, dropout, batch normalization.

### Topic 7 — RAI: Technical Reliability and Explainability
Technical reliability, robustness, explainability, understanding model behaviour.

### Topic 8 — Natural Language Processing
Text preprocessing, tokenization, stop words, Bag-of-Words, TF-IDF, n-grams, Word2Vec, text classification with Naive Bayes, sentiment analysis, named entity recognition.

### Topic 9 — Computer Vision Fundamentals
Image basics, pixels, RGB, resolution, OpenCV, filtering, edge detection, convolution, pooling, data augmentation, transfer learning, object detection, Vision Transformer concepts.

### Topic 10 — Large Language Models (Tech)
Transformer architecture, embeddings, tokenization, self-attention, multi-head attention, positional encoding, encoder vs. decoder, causal attention, pre-training vs. fine-tuning, attention mechanism.

### Topic 11 — Retrieval-Augmented Generation
End-to-end RAG pipeline, document chunking, embedding models, hybrid search, reranking, context-window management, retrieval evaluation.
