# Experiment results (mean ± sd over seeds 0, 1, 2)

## Experiment 2a · Occupation recommendation vs standard recommenders

| Method | MRR (simulated) | MRR (real held-out) | Recall@10 (real held-out) | NDCG@10 (real held-out) |
|---|---|---|---|---|
| Random | 0.148 ± 0.006 | 0.140 ± 0.004 | 0.369 ± 0.018 | 0.165 ± 0.007 |
| Popularity | 0.144 ± 0.000 | 0.208 ± 0.000 | 0.619 ± 0.000 | 0.282 ± 0.000 |
| Jaccard overlap | 0.651 ± 0.011 | 0.308 ± 0.004 | 0.659 ± 0.011 | 0.373 ± 0.005 |
| TF-IDF content-based | 0.771 ± 0.001 | 0.463 ± 0.005 | 0.821 ± 0.003 | 0.540 ± 0.004 |
| Weighted skill overlap | 0.619 ± 0.008 | 0.419 ± 0.020 | 0.812 ± 0.013 | 0.502 ± 0.018 |
| **Node2Vec (ours)** | 0.615 ± 0.004 | 0.558 ± 0.010 | 0.884 ± 0.010 | 0.631 ± 0.010 |

## Experiment 2b · Link prediction (occupation → skill) vs collaborative filtering

| Method | AUC | Hits@10 | MRR |
|---|---|---|---|
| Popularity | 0.840 ± 0.010 | 0.551 ± 0.010 | 0.353 ± 0.033 |
| Item-based CF | 0.799 ± 0.010 | 0.503 ± 0.030 | 0.326 ± 0.017 |
| User-based CF | 0.851 ± 0.011 | 0.542 ± 0.015 | 0.350 ± 0.026 |
| **Node2Vec (ours)** | 0.555 ± 0.012 | 0.063 ± 0.004 | 0.039 ± 0.002 |

## Experiment 1 · Edge-weighting ablation

### Node2Vec

| Weighting | MRR (simulated) | MRR (real held-out) | Recall@10 (real held-out) | LP AUC | LP MRR |
|---|---|---|---|---|---|
| uniform | 0.600 ± 0.007 | 0.485 ± 0.007 | 0.852 ± 0.010 | 0.606 ± 0.014 | 0.040 ± 0.008 |
| confidence_only | 0.616 ± 0.006 | 0.539 ± 0.011 | 0.870 ± 0.006 | 0.600 ± 0.039 | 0.041 ± 0.000 |
| demand_only | 0.623 ± 0.005 | 0.536 ± 0.010 | 0.881 ± 0.010 | 0.569 ± 0.015 | 0.042 ± 0.004 |
| no_freshness | 0.613 ± 0.008 | 0.561 ± 0.015 | 0.893 ± 0.020 | 0.562 ± 0.024 | 0.040 ± 0.005 |
| full | 0.615 ± 0.004 | 0.558 ± 0.010 | 0.884 ± 0.010 | 0.555 ± 0.012 | 0.039 ± 0.002 |

### Weighted skill overlap

| Weighting | MRR (simulated) | MRR (real held-out) | Recall@10 (real held-out) |
|---|---|---|---|
| uniform | 0.599 ± 0.008 | 0.272 ± 0.005 | 0.624 ± 0.009 |
| confidence_only | 0.606 ± 0.004 | 0.323 ± 0.010 | 0.696 ± 0.019 |
| demand_only | 0.616 ± 0.007 | 0.349 ± 0.008 | 0.763 ± 0.006 |
| no_freshness | 0.620 ± 0.009 | 0.418 ± 0.019 | 0.813 ± 0.012 |
| full | 0.619 ± 0.008 | 0.419 ± 0.020 | 0.812 ± 0.013 |

_HGT rows are missing because PyTorch is not installed; run this script on a machine with PyTorch to add them._