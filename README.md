# Identifying Emotional Targets (Synchain ABSA)

This project focuses on identifying emotional targets using Aspect-Based Sentiment Analysis (ABSA), leveraging the **Syntax-Opinion-Sentiment Reasoning Chain (Syn-Chain)** approach.

## Overview

The "Syn-Chain" methodology aims to improve sentiment analysis by explicitly modeling the relationships between:
1.  **Syntax**: The grammatical structure of the sentence.
2.  **Opinion**: The specific expressions of sentiment.
3.  **Sentiment**: The overall polarity (positive, negative, neutral) directed at specific targets.

By chaining these elements, the model can more accurately identify *what* the emotion is directed towards (the target) and *why* (the opinion/syntax).

## Reference & Citation

This work is based on the following research:

> **Aspect-Based Sentiment Analysis with Syntax-Opinion-Sentiment Reasoning Chain**  
> *Rui Fan, Shu Li, Tingting He, Yu Liu*  
> Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025)

If you utilize the original Syn-Chain logic, please cite:

```bibtex
@inproceedings{fan2025aspect,
  author       = {Rui Fan and Shu Li and Tingting He and Yu Liu},
  title        = {Aspect-Based Sentiment Analysis with Syntax-Opinion-Sentiment Reasoning Chain},
  booktitle    = {Proceedings of the 31st International Conference on Computational Linguistics, {COLING} 2025},
  pages        = {3123--3137},
  year         = {2025},
  url          = {https://aclanthology.org/2025.coling-main.210/},
}
```