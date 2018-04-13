import io
import numpy as np
import tensorflow as tf
from hparams import hparams
from librosa import effects
from models import create_model
from text import text_to_sequence
from util import audio


class Synthesizer:
  def load(self, checkpoint_path, model_name='tacotron'):
    print('Constructing model: %s' % model_name)
    inputs = tf.placeholder(tf.float32, [1, None, 127], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
    speaker_ids = tf.placeholder(tf.int32, [1], 'speaker_ids')
    with tf.variable_scope('model') as scope:
      self.model = create_model(model_name, hparams)
      self.model.initialize(inputs, input_lengths, None, None, speaker_ids, None, None)
      self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])

    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)


  def synthesize(self, lab_name):
    lab = np.load(lab_name)
    lab = np.expand_dims(lab, axis=0)
    feed_dict = {
      self.model.inputs: lab,
      self.model.input_lengths: np.asarray([lab.shape[1]], dtype=np.int32),
      # change 0 to 1 or others based on the speaker
      self.model.speaker_ids: np.asarray([2], dtype=np.int32)
    }
    wav, mel_outputs = self.session.run([self.wav_output, self.model.mel_outputs[0]], feed_dict=feed_dict)
    wav = audio.inv_preemphasis(wav)
    _len = audio.find_endpoint(wav)
    wav = wav[:_len]
    _len = audio.find_endpoint(wav)
    wav = wav[:_len]
    mel_output = mel_output[:frames, :]
    out = io.BytesIO()
    audio.save_wav(wav, out)
    return out.getvalue(), mel_outputs
