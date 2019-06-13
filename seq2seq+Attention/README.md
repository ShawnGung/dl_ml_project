# seq2seq Attention

This model is a try to implement a seq2seq+Attention model based on [pytorch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html). The model translates chinese into english and record the attention between target and predicted sequences.

## Run
there are three attention modes, including 'dot', 'general', 'concat'
- python main.py --attn_model 'dot'

## references

https://arxiv.org/abs/1508.04025

https://github.com/fancyerii/deep_learning_theory_and_practice/blob/master/codes/ch05/seq2seq-translation-batched-cn-fixbug.ipynb

https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/masked_cross_entropy.py
