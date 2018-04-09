import numpy as np
import os
import random
import tensorflow as tf
import threading
import time
import traceback
from text import cmudict, text_to_sequence
from util.infolog import log


_batches_per_group = 32
_p_cmudict = 0.5
_pad = 0


class DataFeeder(threading.Thread):
  '''Feeds batches of data into a queue on a background thread.'''

  def __init__(self, coordinator, data_paths, hparams):
    super(DataFeeder, self).__init__()
    self._coord = coordinator
    self._hparams = hparams
    self.data_paths = data_paths
    self.data_path_to_id = {data_path: _id for _id, data_path in enumerate(data_paths)}
    prefixes_dict = {}
    offset_dict = {}
    for data_path in data_paths:
        prefixes = []
        with open(os.path.join(data_path, 'ids.train'), 'r') as fi:
            for line in fi:
                line = line.strip()
                if line:
                    prefixes.append(line)
        prefixes_dict[data_path] = prefixes 
        offset_dict[data_path] = 0
    self._prefixes_dict = prefixes_dict
    self._offset_dict = offset_dict

    self._placeholders = [
      tf.placeholder(tf.float32, [None, None, hparams.num_labs], 'inputs'),
      tf.placeholder(tf.int32, [None], 'input_lengths'),
      tf.placeholder(tf.float32, [None, None, hparams.num_mels], 'mel_targets'),
      tf.placeholder(tf.float32, [None, None, hparams.num_freq], 'linear_targets'),
      tf.placeholder(tf.string, [None], 'prefixes'),
      tf.placeholder(tf.int32, [None], 'speaker_ids'),
      tf.placeholder(tf.int32, [None], 'target_lengths')
    ]

    # Create queue for buffering data:
    queue = tf.FIFOQueue(8, [tf.float32, tf.int32, tf.float32, tf.float32, tf.string, tf.int32, tf.int32], name='input_queue')
    self._enqueue_op = queue.enqueue(self._placeholders)
    self.inputs, self.input_lengths, self.mel_targets, self.linear_targets, self.prefixes, self.speaker_ids, self.target_lengths = queue.dequeue()
    self.inputs.set_shape(self._placeholders[0].shape)
    self.input_lengths.set_shape(self._placeholders[1].shape)
    self.mel_targets.set_shape(self._placeholders[2].shape)
    self.linear_targets.set_shape(self._placeholders[3].shape)
    self.prefixes.set_shape(self._placeholders[4].shape)
    self.speaker_ids.set_shape(self._placeholders[5].shape)
    self.target_lengths.set_shape(self._placeholders[6].shape)

    # Load CMUDict: If enabled, this will randomly substitute some words in the training data with
    # their ARPABet equivalents, which will allow you to also pass ARPABet to the model for
    # synthesis (useful for proper nouns, etc.)
    if hparams.use_cmudict:
      cmudict_path = os.path.join(self._datadir, 'cmudict-0.7b')
      if not os.path.isfile(cmudict_path):
        raise Exception('If use_cmudict=True, you must download ' +
          'http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b to %s'  % cmudict_path)
      self._cmudict = cmudict.CMUDict(cmudict_path, keep_ambiguous=False)
      log('Loaded CMUDict with %d unambiguous entries' % len(self._cmudict))
    else:
      self._cmudict = None


  def start_in_session(self, session):
    self._session = session
    self.start()


  def run(self):
    try:
      while not self._coord.should_stop():
        self._enqueue_next_group()
    except Exception as e:
      traceback.print_exc()
      self._coord.request_stop(e)


  def _enqueue_next_group(self):
    start = time.time()

    # Read a group of examples:
    n = self._hparams.batch_size
    r = self._hparams.outputs_per_step
    examples = []
    for data_path in self.data_paths:
        example = [self._get_next_example(data_path) for i in range(int(n * _batches_per_group // len(self.data_paths)))]
        examples.extend(example)
        
    # Bucket examples based on similar output sequence length for efficiency:
    examples.sort(key=lambda x: x[-1])
    batches = [examples[i:i+n] for i in range(0, len(examples), n)]
    random.shuffle(batches)

    log('Generated %d batches of size %d in %.03f sec' % (len(batches), n, time.time() - start))
    for batch in batches:
      feed_dict = dict(zip(self._placeholders, _prepare_batch(batch, r)))
      self._session.run(self._enqueue_op, feed_dict=feed_dict)


  def _get_next_example(self, data_path):
    '''Loads a single example (input, mel_target, linear_target, cost) from disk'''
    if self._offset_dict[data_path] >= len(self._prefixes_dict[data_path]):
        self._offset_dict[data_path] = 0
        random.shuffle(self._prefixes_dict[data_path])
    prefix = self._prefixes_dict[data_path][self._offset_dict[data_path]]
    self._offset_dict[data_path] += 1
    label = np.load(os.path.join(data_path, 'np_lab', prefix + '_lab.npy'))
    mel_target = np.load(os.path.join(data_path, 'mel', prefix + '_mel.npy'))
    linear_target = np.load(os.path.join(data_path, 'spec', prefix + '_spec.npy'))
    return (label, mel_target, linear_target, prefix, self.data_path_to_id[data_path], len(linear_target))

  def _maybe_get_arpabet(self, word):
    arpabet = self._cmudict.lookup(word)
    return '{%s}' % arpabet[0] if arpabet is not None and random.random() < 0.5 else word


def _prepare_batch(batch, outputs_per_step):
  random.shuffle(batch)
  inputs = _prepare_inputs([x[0] for x in batch])
  input_lengths = np.asarray([len(x[0]) for x in batch], dtype=np.int32)
  mel_targets = _prepare_targets([x[1] for x in batch], outputs_per_step)
  linear_targets = _prepare_targets([x[2] for x in batch], outputs_per_step)
  prefixes = [x[3] for x in batch]
  speaker_ids = np.asarray([x[4] for x in batch], dtype=np.int32)
  target_lengths = np.asarray([x[5] for x in batch], dtype=np.int32)
  return (inputs, input_lengths, mel_targets, linear_targets, prefixes, speaker_ids, target_lengths)


def _prepare_inputs(inputs):
  max_len = max((len(x) for x in inputs))
  return np.stack([_pad_input(x, max_len) for x in inputs])


def _prepare_targets(targets, alignment):
  max_len = max((len(t) for t in targets)) + 1
  return np.stack([_pad_target(t, _round_up(max_len, alignment)) for t in targets])


def _pad_input(x, length):
  return np.pad(x, [(0, length - x.shape[0]), (0,0)], mode='constant', constant_values=_pad)


def _pad_target(t, length):
  return np.pad(t, [(0, length - t.shape[0]), (0,0)], mode='constant', constant_values=_pad)


def _round_up(x, multiple):
  remainder = x % multiple
  return x if remainder == 0 else x + multiple - remainder
