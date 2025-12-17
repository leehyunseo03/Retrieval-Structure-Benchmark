# Retrieval Structure Benchmark 

**Genifier : Custom Retrieval Structure for Genipai**

### M3DocVQA (Text Only)
> **30 pdfs & 30 questions(1 question for 1 pdf)**

- Property Graph Index(LLM) + Normal Parser
    | K | Recall | MRR |
    | --- | --- | --- |
    | 1 | 0.5862 | 0.5862 |
    | 3 | 0.8276 | 0.6897 |
    | 5 | 0.8966 | 0.7034 |
    | 10 | 0.9310 | 0.7084 |

- Property Graph Index(Implicit) + Normal Parser + Reranker(cross-encoder/ms-marco-MiniLM-L-6-v2)
    | K | Recall | MRR |
    | --- | --- | --- |
    | 1 | 0.7879 | 7879 |
    | 3 | 0.9394 | 0.8586 |
    | 5 | 0.9394 | 0.8586 |
    | 10 | 0.9394 | 0.8586 |

### QASPER (Text Only)
> **30 pdfs & 93 questions**

- Property Graph Index(LLM) + Normal Parser
    | K | Recall | MRR |
    | --- | --- | --- |
    | 1 | 0.3034 | 0.3034 |
    | 3 | 0.4944 | 0.3858 |
    | 5 | 0.5618 | 0.4009 |
    | 10 | 0.6629 | 0.4149 |

- Vector+BM25(RRF) + Normal Parser + Reranker(cross-encoder/ms-marco-MiniLM-L-6-v2)    
    | K | Recall | MRR |
    | --- | --- | --- |
    | 1 | 0.5238 | 0.5238 |
    | 3 | 0.6071 | 0.5615 |
    | 5 | 0.6667 | 0.5758 |
    | 10 | 0.7143 | 0.5818 |
    
- Property Graph Index(Implicit) + Normal Parser + Reranker(cross-encoder/ms-marco-MiniLM-L-6-v2)
    | K | Recall | MRR |
    | --- | --- | --- |
    | 1 | 0.5393 | 0.5393 |
    | 3 | 0.6742 | 0.6030 |
    | 5 | 0.7416 | 0.6187 |
    | 10 | 0.7753 | 0.6234 |

### FinanceBench (Text Only)
> **5 pdfs & 8 questions**

- Property Graph Index(LLM) + Normal Parser
    | K | Recall | MRR |
    | --- | --- | --- |
    | 1 | 0.6250 | 0.6250 |
    | 3 | 0.7500 | 0.6875 |
    | 5 | 0.7500 | 0.6875 |
    | 10 | 0.8750 | 0.7000 |

- Property Graph Index(Implicit) + Normal Parser + Reranker(cross-encoder/ms-marco-MiniLM-L-6-v2)
    | K | Recall | MRR |
    | --- | --- | --- |
    | 1 | 0.4068 | 0.4068 |
    | 3 | 0.6441 | 0.5028 |
    | 5 | 0.6610 | 0.5071 |
    | 10 | 0.7119 | 0.5140 |