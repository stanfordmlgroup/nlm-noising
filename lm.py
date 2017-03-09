import os
import time
import copy
from os.path import join as pjoin
from six.moves import xrange

import numpy as np
import tensorflow as tf
from loader import TextLoader, noise_batch

from cfg import PTB_DATA_PATHS, TEXT8_DATA_PATHS
from opt import get_optimizer

import logging
import sys
logging.basicConfig(level=logging.INFO)

flags = tf.flags

# Settings
flags.DEFINE_integer("hidden_dim", 512, "hidden dimension")
flags.DEFINE_integer("layers", 2, "number of hidden layers")
flags.DEFINE_integer("unroll", 35, "number of time steps to unroll for BPTT")
flags.DEFINE_integer("batch_size", 20, "batch size")
flags.DEFINE_float("init_scale", 0.1, "scale for random initialization")
flags.DEFINE_float("learning_rate", 1.0, "initial learning rate")
flags.DEFINE_float("learning_rate_decay", 0.5, "amount to decrease learning rate")
flags.DEFINE_float("decay_threshold", 0.0, "decrease learning rate if validation cost difference less than this value")
flags.DEFINE_integer("max_decays", 8, "stop decreasing learning rate after this many times")
flags.DEFINE_float("drop_prob", 0.0, "probability of dropping units")
flags.DEFINE_float("gamma", 0.0, "probability of noising input data")
flags.DEFINE_boolean("absolute_discounting", False, "scale gamma by absolute discounting factor")
flags.DEFINE_integer("max_epochs", 400, "maximum number of epochs to train")
flags.DEFINE_float("clip_norm", 5.0, "value at which to clip gradients")
flags.DEFINE_string("optimizer", "sgd", "optimizer")
flags.DEFINE_string("run_dir", "sandbox", "directory to store experiment outputs")
flags.DEFINE_string("token_type", "word", "use word or character tokens")
flags.DEFINE_string("scheme", "blank", "use blank or ngram noising scheme")
flags.DEFINE_string("ngram_scheme", "unigram", "use {unigram, uniform, bgkn, mbgkn}")
flags.DEFINE_string("restore_checkpoint", None, "checkpoint file to restore model parameters from")
flags.DEFINE_integer("seed", 123, "random seed to use")
flags.DEFINE_integer("steps_per_summary", 10, "how many steps between writing summaries")
flags.DEFINE_boolean("final", False, "final evaluation (run on test after picked best model)")
flags.DEFINE_string("dataset", "ptb", "ptb or text8")

FLAGS = flags.FLAGS

# Getting stale file handle errors
def log_info(s):
  try:
    logging.info(s)
  except IOError:
    time.sleep(60)

class LanguageModel(object):

  def __init__(self, flags, vocab_size, is_training=True):
    batch_size = flags.batch_size
    unroll = flags.unroll
    self._x = tf.placeholder(tf.int32, [batch_size, unroll])
    self._y = tf.placeholder(tf.int32, [batch_size, unroll])
    self._len = tf.placeholder(tf.int32, [None, ])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(flags.hidden_dim, forget_bias=1.0, state_is_tuple=True)
    if is_training and flags.drop_prob > 0:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=1.0-flags.drop_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * flags.layers, state_is_tuple=True)
    self._initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      self.embeddings = tf.get_variable("embeddings", [vocab_size, flags.hidden_dim])
      inputs = tf.nn.embedding_lookup(self.embeddings, self._x)
    if is_training and flags.drop_prob > 0:
      inputs = tf.nn.dropout(inputs, 1.0 - flags.drop_prob)

    # These options (fixed unroll or dynamic_rnn) should give same results but
    # using fixed here since faster
    if True:
      outputs = []
      state = self._initial_state
      with tf.variable_scope("RNN"):
        for time_step in range(unroll):
          if time_step > 0: tf.get_variable_scope().reuse_variables()
          (cell_output, state) = cell(inputs[:, time_step, :], state)
          outputs.append(cell_output)
      outputs = tf.reshape(tf.concat(1, outputs), [-1, flags.hidden_dim])
    else:
      with tf.variable_scope("RNN"):
          outputs, state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=self._len,
              initial_state=self._initial_state, dtype=tf.float32, time_major=False)
      outputs = tf.reshape(outputs, [-1, flags.hidden_dim])

    softmax_w = tf.get_variable("softmax_w", [flags.hidden_dim, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(outputs, softmax_w) + softmax_b
    seq_loss = tf.nn.seq2seq.sequence_loss_by_example(
      [tf.reshape(logits, [-1, vocab_size])],
      [tf.reshape(self._y, [-1])],
      [tf.ones([batch_size * unroll])])
    self.loss = tf.reduce_sum(seq_loss) / batch_size
    self._final_state = state

    if not is_training:
      return

    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    shapes = [tvar.get_shape() for tvar in tvars]
    log_info("# params: %d" % np.sum([np.prod(s) for s in shapes]))
    grads = tf.gradients(self.loss, tvars)
    if flags.clip_norm is not None:
      grads, grads_norm = tf.clip_by_global_norm(grads, flags.clip_norm)
    else:
      grads_norm = tf.global_norm(grads)
    optimizer = get_optimizer(flags.optimizer)(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    # Summaries for TensorBoard, note this is only within training portion
    with tf.name_scope("summaries"):
      tf.scalar_summary("loss", self.loss / unroll)
      tf.scalar_summary("learning_rate", self.lr)
      tf.scalar_summary("grads_norm", grads_norm)

  def set_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

def run_epoch(epoch_ind, session, model, loader, split, update_op, flags,
        writer=None, summary_op=None, verbose=True):
  """Run an epoch of training/testing"""
  epoch_size = loader.get_num_batches(split)
  start_time = time.time()
  total_cost = 0.0
  state = session.run(model._initial_state)
  iters = 0
  for k in xrange(epoch_size):
    x, y = loader.get_batch(split, k)
    if split == "train":
      gamma = flags.gamma
      x, y = noise_batch(x, y, flags, loader, gamma=gamma)
    seq_len = [y.shape[1]] * flags.batch_size
    fetches = [model.loss, update_op, model._final_state]
    feed_dict = {model._x: x,
                 model._y: y,
                 model._len: seq_len,
                 model._initial_state: state}
    if summary_op is not None and writer is not None:
      fetches = [summary_op] + fetches
      summary, cost, _, state = session.run(fetches, feed_dict)
      if k % flags.steps_per_summary == 0:
        writer.add_summary(summary, epoch_size*epoch_ind + k)
    else:
      cost, _, state = session.run(fetches, feed_dict)
    total_cost += cost
    iters += flags.unroll

    if k % (epoch_size // 10) == 10 and verbose:
      log_info("%.3f perplexity: %.3f speed: %.0f tps" %
        (k * 1.0 / epoch_size, np.exp(total_cost / iters),
        iters * flags.batch_size / (time.time() - start_time)))

  return np.exp(total_cost / iters)

def main(_):
  if not os.path.exists(FLAGS.run_dir):
    os.makedirs(FLAGS.run_dir)
  file_handler = logging.FileHandler("{0}/log.txt".format(FLAGS.run_dir))
  logging.getLogger().addHandler(file_handler)

  DATA_PATHS = PTB_DATA_PATHS if FLAGS.dataset == "ptb" else TEXT8_DATA_PATHS
  log_info(str(DATA_PATHS))
  data_loader = TextLoader(DATA_PATHS, FLAGS.batch_size, FLAGS.unroll,
          FLAGS.token_type)
  vocab_size = len(data_loader.token_to_id)
  log_info("Vocabulary size: %d" % vocab_size)
  log_info(FLAGS.__flags)

  eval_flags = copy.deepcopy(FLAGS)
  eval_flags.batch_size = 1
  eval_flags.unroll = 1

  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale, seed=FLAGS.seed)

    # Create training, validation, and evaluation models
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      mtrain = LanguageModel(FLAGS, vocab_size, is_training=True)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      mvalid = LanguageModel(FLAGS, vocab_size, is_training=False)
      mtest = LanguageModel(eval_flags, vocab_size, is_training=False)

    summary_op = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.run_dir)
    model_saver = tf.train.Saver(max_to_keep=FLAGS.max_epochs)
    tf.initialize_all_variables().run()

    if FLAGS.restore_checkpoint is not None:
      saver.restore(session, FLAGS.restore_checkpoint)

    lr = FLAGS.learning_rate
    decay_count = 0
    prev_valid_perplexity = None
    valid_perplexities = list()

    for k in xrange(FLAGS.max_epochs):
      mtrain.set_lr(session, lr)
      log_info("Epoch %d, learning rate %f" % (k, lr))

      train_perplexity = run_epoch(k, session, mtrain, data_loader, "train",
          mtrain._train_op, FLAGS, writer=train_writer, summary_op=summary_op)
      log_info("Epoch: %d Train Perplexity: %.3f" % (k, train_perplexity))
      valid_perplexity = run_epoch(k, session, mvalid, data_loader, "valid",
          tf.no_op(), FLAGS, verbose=False)
      log_info("Epoch: %d Valid Perplexity: %.3f" % (k, valid_perplexity))
      if prev_valid_perplexity != None and\
              np.log(best_valid_perplexity) - np.log(valid_perplexity) < FLAGS.decay_threshold:
        lr = lr * FLAGS.learning_rate_decay
        decay_count += 1
        log_info("Loading epoch %d parameters, perplexity %f" %\
                (best_epoch, best_valid_perplexity))
        model_saver.restore(session, pjoin(FLAGS.run_dir, "model_epoch%d.ckpt" % best_epoch))
      prev_valid_perplexity = valid_perplexity

      valid_perplexities.append(valid_perplexity)
      if valid_perplexity <= np.min(valid_perplexities):
        best_epoch = k
        best_valid_perplexity = valid_perplexities[best_epoch]
        save_path = model_saver.save(session, pjoin(FLAGS.run_dir, "model_epoch%d.ckpt" % k))
        log_info("Saved model to file: %s" % save_path)

      if decay_count > FLAGS.max_decays:
        log_info("Reached maximum number of decays, quiting after epoch %d" % k)
        break

    log_info("Loading epoch %d parameters, perplexity %f" %\
            (best_epoch, best_valid_perplexity))
    model_saver.restore(session, pjoin(FLAGS.run_dir, "model_epoch%d.ckpt" % best_epoch))
    data_loader = TextLoader(DATA_PATHS, eval_flags.batch_size, eval_flags.unroll, FLAGS.token_type)
    if FLAGS.final:
      test_perplexity = run_epoch(k, session, mtest, data_loader, "test",
        tf.no_op(), eval_flags, verbose=False)
      log_info("Test Perplexity: %.3f" % test_perplexity)

if __name__ == "__main__":
  tf.app.run()
