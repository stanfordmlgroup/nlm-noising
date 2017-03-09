import collections
import os

import nltk
import numpy as np
import bisect

from utils import bigram_counts, trigram_counts, build_continuations
from utils import estimate_modkn_discounts


"""
Adapted from
- https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
"""


def _read_tokens(filename, level="word"):
  with open(filename, "r") as f:
    if "ptb" in filename:
      tokens = f.read().replace("\n", "<eos>")
    elif "text8" in filename:
      tokens = f.read().strip()
    else:
      assert(False)
    if level == "word":
      tokens = tokens.split()
    return tokens


def _file_to_token_ids(filename, token_to_id, level):
  data = _read_tokens(filename, level=level)
  return data, [token_to_id[token] for token in data]


def _build_vocab(filename, level):
  data = _read_tokens(filename, level=level)
  counter = collections.Counter(data)
  # Use this to get tokens sorted by frequencies
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  total_count = sum(counter.values())
  frequencies = dict((k, v / float(total_count))
                     for k, v in counter.iteritems())

  # Compute number of different histories
  bg_hist_sets = collections.defaultdict(set)
  for k in xrange(1, len(data)):
    bg_hist_sets[data[k]].add(data[k - 1])
  bg_hist_counts = dict([(k, len(s)) for k, s in bg_hist_sets.iteritems()])
  # NOTE Edge case here where first word never appears again
  if data[0] not in bg_hist_counts:
    bg_hist_counts[data[0]] = 1
  total_hists = sum(bg_hist_counts.values())

  tokens, _ = list(zip(*count_pairs))
  token_to_id = dict(zip(tokens, range(len(tokens))))
  sorted_frequencies = [frequencies[token] for token in tokens]
  sorted_hist_freqs = [bg_hist_counts[token] /
                       float(total_hists) for token in tokens]

  return token_to_id, sorted_frequencies, sorted_hist_freqs


def _reshape_data(raw_data, batch_size, unroll):
  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
  epoch_size = batch_len // unroll
  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or unroll")
  return data


def load_text_data(data_paths, level):
  assert len(data_paths) == 3
  train_path = data_paths[0]
  valid_path = data_paths[1]
  test_path = data_paths[2]

  token_to_id, frequencies, hist_freqs = _build_vocab(train_path, level)
  train_tokens, train_data = _file_to_token_ids(train_path, token_to_id, level)
  _, valid_data = _file_to_token_ids(valid_path, token_to_id, level)
  _, test_data = _file_to_token_ids(test_path, token_to_id, level)

  return train_data, valid_data, test_data, token_to_id, frequencies, hist_freqs, train_tokens


def sample_index(ps_cumsum):
  return bisect.bisect(ps_cumsum, np.random.random() * ps_cumsum[-1])


def noise_batch(x, y, flags, loader, gamma=0.0, wmat=None):
  if gamma == 0.0:
    return x, y
  continuations = loader.continuations
  x_, y_ = np.array(x), np.array(y)
  for row in xrange(x.shape[0]):
    for col in xrange(x.shape[1]):
      if flags.absolute_discounting:
        #if col < 1:
        if False:
          p = 0
        else:
          context = list()
          #context.append(loader.id_to_token[x[row, col-1]])
          context.append(loader.id_to_token[x[row, col]])
          # Can also compute D = n1/(n1+n2) as described in Chen & Goodman
          total, distinct = continuations["total"][tuple(context)],\
              continuations["distinct"][tuple(context)]
          if flags.ngram_scheme != "mbgkn":
            p = (gamma / total) * distinct
          else:
            p = gamma * (loader.D1 * loader.N1_lookup[context[0]] +
                         loader.D2 * loader.N2_lookup[context[0]] +
                         loader.D3p * loader.N3p_lookup[context[0]]) / float(total)
      else:
        p = gamma
      draw = np.random.binomial(1, p)
      if draw:
        if flags.scheme == "blank":
          x_[row, col] = loader.token_to_id['<_>']
        elif flags.scheme == "ngram":
          if flags.ngram_scheme == "unigram":
            freqs_cumsum = loader.frequencies_cumsum
          elif "kn" in flags.ngram_scheme:
            pass
          else:
            assert False

          if "kn" not in flags.ngram_scheme:
            x_[row, col] = sample_index(freqs_cumsum)
          else:
            x_[row, col] = sample_index(loader.hist_freqs_cumsum)
            y_[row, col] = sample_index(loader.hist_freqs_cumsum)
        else:
          raise
  return x_, y_


class TextLoader(object):

  def __init__(self, data_paths, batch_size, unroll, level):
    self.batch_size = batch_size
    self.unroll = unroll
    train_data, valid_data, test_data, token_to_id, frequencies, hist_freqs, train_tokens = load_text_data(
        data_paths, level)
    self.bg_counts = bigram_counts(train_tokens)
    self.tg_counts = trigram_counts(train_tokens)
    self.token_to_id = token_to_id
    # NOTE extends the vocabulary
    self.token_to_id['<_>'] = len(self.token_to_id)
    self.id_to_token = dict((v, k) for k, v in self.token_to_id.iteritems())
    train_data = _reshape_data(train_data, batch_size, unroll)
    valid_data = _reshape_data(valid_data, batch_size, unroll)
    test_data = _reshape_data(test_data, batch_size, unroll)
    self.split_data = {"train": train_data, "valid": valid_data,
                       "test": test_data}
    self.frequencies = frequencies
    self.frequencies_cumsum = np.cumsum(frequencies)
    self.hist_freqs = hist_freqs
    self.hist_freqs_cumsum = np.cumsum(hist_freqs)
    self.continuations = build_continuations(self.bg_counts)
    bgs = nltk.bigrams(train_tokens)
    if level == "word":
      self.D1, self.D2, self.D3p, self.N1_lookup, self.N2_lookup, self.N3p_lookup = estimate_modkn_discounts(
          bgs)

  def get_num_batches(self, split):
    return (self.split_data[split].shape[1] - 1) // self.unroll

  def get_batch(self, split, index):
    split_data = self.split_data[split]
    i = index
    x = split_data[:, i * self.unroll:(i + 1) * self.unroll]
    y = split_data[:, i * self.unroll + 1:(i + 1) * self.unroll + 1]
    return x, y

if __name__ == "__main__":
  from cfg import PTB_DATA_PATHS
  loader = TextLoader(PTB_DATA_PATHS, 20, 35, "word")

  print("most frequent token: %s" %
        loader.id_to_token[np.argmax(loader.frequencies)])
  print("token with most distinct histories: %s" %
        loader.id_to_token[np.argmax(loader.hist_freqs)])
  print("tokens with most distinct continuations: %s" % sorted(loader.continuations[
        "distinct"].iterkeys(), key=(lambda key: -loader.continuations["distinct"][key]))[0:10])
  print("tokens with most total continuations: %s" % sorted(loader.continuations[
        "total"].iterkeys(), key=(lambda key: -loader.continuations["total"][key]))[0:10])
