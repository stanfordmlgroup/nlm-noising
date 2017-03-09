import nltk
from collections import Counter
from collections import defaultdict


# n-gram stuff


def bigram_counts(word_list):
  bgs = nltk.bigrams(word_list)
  fdist = nltk.FreqDist(bgs)
  d = Counter()
  for k, v in fdist.items():
    d[k] = v
  return d


def trigram_counts(word_list):
  tgs = nltk.trigrams(word_list)
  fdist = nltk.FreqDist(tgs)
  d = Counter()
  for k, v in fdist.items():
    d[k] = v
  return d


def build_continuations(counts_dict):
  total = defaultdict(int)
  distinct = defaultdict(int)
  for key in counts_dict:
    context = key[:-1]
    total[context] += counts_dict[key]
    distinct[context] += 1
  return {"total": total, "distinct": distinct}


def estimate_modkn_discounts(ngrams):
  # Get counts
  counts = Counter(ngrams)
  N1 = float(len([k for k in counts if counts[k] == 1]))
  N2 = float(len([k for k in counts if counts[k] == 2]))
  N3 = float(len([k for k in counts if counts[k] == 3]))
  N4 = float(len([k for k in counts if counts[k] == 4]))
  N3p = float(len([k for k in counts if counts[k] >= 3]))

  # Estimate discounting parameters
  Y = N1 / (N1 + 2 * N2)
  D1 = 1 - 2 * Y * (N2 / N1)
  D2 = 2 - 3 * Y * (N3 / N2)
  D3p = 3 - 4 * Y * (N4 / N3)

  # FIXME(zxie) Assumes bigrams for now
  # Also compute N1/N2/N3p lookups (context -> n-grams with count 1/2/3+)
  N1_lookup = Counter()
  N2_lookup = Counter()
  N3p_lookup = Counter()
  for bg in counts:
    if counts[bg] == 1:
      N1_lookup[bg[0]] += 1
    elif counts[bg] == 2:
      N2_lookup[bg[0]] += 1
    else:
      N3p_lookup[bg[0]] += 1

  return D1, D2, D3p, N1_lookup, N2_lookup, N3p_lookup
