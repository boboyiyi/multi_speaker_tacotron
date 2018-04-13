# multi_speaker_tacotron

## Quick Start

### Installing dependencies

1. Running environment: tensorflow 1.4.0 + python 3.6

2. pip install -r requirements.txt

### Training

#### Preprocess the data

```
python preprocess.py --dataset [your dataset]
```

#### Train a model

Before running, you need to change the default configuration of hparams.py according to your dataset.

```
sh run_train.sh
```

#### Synthesize from a checkpoint

```
sh run_eval.sh
```

## References:

- https://github.com/keithito/tacotron
- https://github.com/kastnerkyle/multi-speaker-tacotron-tensorflow

## To do:

- [x] Upload source code.
- [x] Run and verify.
- [ ] Detailed readme.