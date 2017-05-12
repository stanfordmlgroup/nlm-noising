## Data Noising as Smoothing in Neural Network Language Models

Dependencies
- [Tensorflow](https://github.com/tensorflow/tensorflow) (Tested with v1.1)
- [NLTK](http://www.nltk.org/)

### Overview

Based off of Tensorflow inplementation [here](https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py),
which is in turn based off of PTB LSTM implementation [here](https://github.com/wojzaremba/lstm).

Implements noising for neural language modeling as described in this [paper](https://arxiv.org/abs/1703.02573).
```
@inproceedings{noising2017,
  title={Data Noising as Smoothing in Neural Network Language Models},
  author={Xie, Ziang and Wang, Sida I. and Li, Jiwei and L{\'e}vy, Daniel and Nie, Aiming and Jurafsky, Dan and Ng, Andrew Y.},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}
```
The noising code can be found in `loader.py` and `utils.py`.

### How to run

First download PTB data from [here](https://github.com/wojzaremba/lstm/tree/master/data)
and put in data directory. Make sure to update paths in `cfg.py` to point to data.
Alternatively, you can also grab the Text8 data [here](http://mattmahoney.net/dc/text8.zip), then run
the script `data/text8/makedata-text8.sh`.

Then run `lm.py`. Here's an example setting:

```bash
python lm.py --run_dir /tmp/lm_1500_kn  --hidden_dim 1500 --drop_prob 0.65 --gamma 0.2 --scheme ngram --ngram_scheme kn --absolute_discounting
```
