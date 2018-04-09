import argparse
import os
import re
import numpy as np
from hparams import hparams, hparams_debug_string
from synthesizer_lab import Synthesizer

def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  synth = Synthesizer()
  synth.load(args.checkpoint)
  base_path = get_output_base_path(args.checkpoint)
  with open(os.path.join(args.data_path, 'ids.test'), 'r') as fi:
    for line in fi:
      line = line.strip()
      if line:
        lab_name = os.path.join(args.data_path, 'np_lab', line + '_lab.npy')
        test_path = os.path.join(args.data_path, 'test')
        if not os.path.exists(test_path):
          os.mkdir(test_path) 
        wav_name = os.path.join(test_path, line + '.wav')
        mel_name = os.path.join(test_path, line + '_mel.npy')
        with open(wav_name, 'wb') as f:
          out, mel_outputs = synth.synthesize(lab_name)
          f.write(out)
          mel_outputs = np.squeeze(mel_outputs, axis=0)
          np.save(mel_name, mel_outputs)
          # f.write(synth.synthesize(lab_name))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--data_path', required=True, default='./datasets/LJ')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
