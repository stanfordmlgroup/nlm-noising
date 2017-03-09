import tensorflow as tf

def get_optimizer(name):
  if name == "sgd":
    return tf.train.GradientDescentOptimizer
  elif name == "adam":
    return tf.train.AdamOptimizer
  else:
    assert False
