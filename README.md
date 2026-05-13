# AGRank

Official implementation of **AGRank: Automatic Keyphrase Extraction with Attentive Graph Convolutional Networks**.

## Overview

AGRank is an unsupervised keyphrase extraction framework that combines Transformer-based attention mechanisms with graph-based ranking methods to better capture semantic relationships between phrases.

The method constructs phrase-level graphs using attention-derived relationships from pretrained language models and refines phrase importance estimation through graph algorithms. AGRank is designed to improve both local semantic understanding and global structural modeling in keyphrase extraction tasks.

## Features

- Unsupervised keyphrase extraction
- Attention-enhanced graph construction
- Graph-based phrase ranking
- Compatible with pretrained Transformer models
- Improved modeling of phrase relationships
- Benchmark evaluation support

## Additional Notes

During the execution of `005_get_base_graph`, the pretrained Sent2Vec model `wiki_bigrams.bin` (approximately 17GB) may need to be downloaded manually from:

https://s3.amazonaws.com/epfml-sent2vec-models/wiki_bigrams.bin

```bibtex
@inproceedings{ding2022agrank,
  title={AGRank: Augmented graph-based unsupervised keyphrase extraction},
  author={Ding, Haoran and Luo, Xiao},
  booktitle={Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  pages={230--239},
  year={2022}
}
```
