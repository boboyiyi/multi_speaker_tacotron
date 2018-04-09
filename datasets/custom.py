from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from util import audio


def build_from_path(in_dir, num_workers=1, tqdm=lambda x: x):
  '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.

    Args:
      in_dir: The directory where you have downloaded the LJ Speech dataset
      out_dir: The directory to write the output into
      num_workers: Optional number of worker processes to parallelize across
      tqdm: You can optionally pass tqdm to get a nice progress bar

    Returns:
      A list of tuples describing the training examples. This should be written to train.txt
  '''

  # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
  # can omit it and just call _process_utterance on each input if you want.
  if not os.path.exists(os.path.join(in_dir, 'spec')):
    os.mkdir(os.path.join(in_dir, 'spec'))
  if not os.path.exists(os.path.join(in_dir, 'mel')):
    os.mkdir(os.path.join(in_dir, 'mel'))
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []
  index = 1
  with open(os.path.join(in_dir, 'ids'), 'r') as f:
    for line in f:
      line = line.strip()
      if line:
        wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % line)
      futures.append(executor.submit(partial(_process_utterance, in_dir, index, wav_path)))
      index += 1
  return [future.result() for future in tqdm(futures)]


def _process_utterance(in_dir, index, wav_path):
  '''Preprocesses a single utterance audio/text pair.

  This writes the mel and linear scale spectrograms to disk and returns a tuple to write
  to the train.txt file.

  Args:
    out_dir: The directory to write the spectrograms into
    index: The numeric index to use in the spectrogram filenames.
    wav_path: Path to the audio file containing the speech input
    text: The text spoken in the input audio file

  Returns:
    A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
  '''

  # Load the audio to a numpy array:
  wav = audio.load_wav(wav_path)

  # Compute the linear-scale spectrogram from the wav:
  spectrogram = audio.spectrogram(wav).astype(np.float32)
  n_frames = spectrogram.shape[1]

  # Compute a mel-scale spectrogram from the wav:
  mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

  # Write the spectrograms to disk:
  name = wav_path.split('/')[-1]
    
  mel_filename = os.path.splitext(name)[0] + '_mel.npy'
  spectrogram_filename = os.path.splitext(name)[0] + '_spec.npy'
  np.save(os.path.join(in_dir, 'mel', mel_filename), mel_spectrogram.T, allow_pickle=False)
  np.save(os.path.join(in_dir, 'spec', spectrogram_filename), spectrogram.T, allow_pickle=False)

  # Return a tuple describing this training example:
  return (spectrogram_filename, mel_filename, n_frames)
