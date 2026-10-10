Document: LLM2Emb_A_two-stage-LLM-based-embedding-framework-for-long-tail_sequential_recommendation.pdf (28 pages)
Questions: 25  |  top-3 chunks per question

| Method | Hit@3 | MRR |
|---|---|---|
| FAISS only | 16/25 = 0.640 | 0.533 |
| BM25 only | 16/25 = 0.640 | 0.547 |
| Hybrid (RRF k=60) | 20/25 = 0.800 | 0.693 |

Per-question results (page = PDF page index, 1-based):

| # | Type | Page | FAISS only | BM25 only | Hybrid (RRF k=60) | Question |
|---|---|---|---|---|---|---|
| 1 | keyword | 1 | miss | hit | miss | According to the abstract, on which low-quality dataset do the improvements in NDCG@k and HR@k exceed 10%? |
| 2 | keyword | 2 | miss | hit | hit | Why is LLM2Emb feasible for real-time recommendation, given that LLM embeddings are computed offline and cached? |
| 3 | keyword | 6 | hit | hit | hit | How does ACCT avoid data leakage when training the LightGCN CF model? |
| 4 | keyword | 7 | hit | miss | hit | What is the role of the lambda_collab hyperparameter in the overall ACCT objective? |
| 5 | keyword | 8 | hit | hit | hit | Into which projection layers of the attention module are the low-rank adaptation matrices inserted during ACCT training? |
| 6 | keyword | 9 | hit | hit | hit | How many negative items are randomly sampled for each user at validation and testing time? |
| 7 | keyword | 10 | miss | miss | miss | How many popularity buckets are used in the Pop-GAR stage? |
| 8 | keyword | 13 | hit | hit | hit | How large is the million-scale interaction dataset used in the efficiency analysis? |
| 9 | keyword | 17 | hit | hit | hit | Which value of k, the number of collaboratively retrieved item pairs, gives the best performance in most cases? |
| 10 | keyword | 22 | miss | hit | hit | What limitation is noted about the popularity gate at inference time? |
| 11 | paraphrased | 3 | miss | miss | hit | How does the paper's neighbor lookup differ from fetching documents at query time when answering questions with an LLM? |
| 12 | paraphrased | 4 | hit | hit | hit | Which way of combining ID vectors with extra item features keeps the values purely ID-based by injecting the extra data only when attention scores are computed? |
| 13 | paraphrased | 8 | hit | hit | hit | What input goes into the sigmoid that produces the mixing weight between the text-derived and ID-derived item vectors? |
| 14 | paraphrased | 11 | hit | hit | hit | On the restaurant-review dataset, which backbone shows the largest improvement over the strongest competitor for overall NDCG@10? |
| 15 | paraphrased | 12 | hit | hit | hit | What lets the method extract dependable meaning even when item metadata is poor? |
| 16 | paraphrased | 14 | hit | miss | hit | If the contrastive adaptation of the language model is skipped and raw pretrained vectors are used, how much can accuracy fall? |
| 17 | paraphrased | 15 | miss | miss | miss | How does the weight given to the text-derived vector change as items get more interactions? |
| 18 | paraphrased | 16 | miss | miss | hit | Which of the tested language models scored a bit below the others? |
| 19 | paraphrased | 19 | hit | miss | hit | What flaw of the graph-convolution collaborative model makes its chosen neighbors noisy for rarely-seen items? |
| 20 | paraphrased | 21 | hit | miss | miss | What did the 2-D projection show about how items are arranged in the embedding space after the proposed tuning? |
| 21 | multi-term | 10 | hit | hit | hit | In the Pop-GAR stage, what are the batch size and the ID embedding dimension? |
| 22 | multi-term | 6 | hit | hit | hit | Which two operations are applied to an item's text to create its two semantically similar views, and what is the drop rate applied to? |
| 23 | multi-term | 17 | hit | hit | hit | In the retrieval-threshold table, what thresholds are used for cosine similarity with the SAS and GCN backbones? |
| 24 | multi-term | 18 | miss | hit | hit | In the grid search over g_max and g_min, which settings yield the best or near-best HR@k and NDCG@k? |
| 25 | multi-term | 20 | miss | miss | miss | What does the text say about how well LightGCN performs inside the framework, and which more powerful collaborative model such as SASRec could give further gains? |
