## Forked from https://github.com/cffan/neural_seq_decoder

Main branch contains best performing implementation, other branches contain some of the other experiments we conducted.

Compared to the original repository (https://github.com/cffan/neural_seq_decoder), also requires `lhotse` library:
> Gets installed automatically with `pip install -e .`  
> Or, can be installed separately with `pip install lhotse`

## Pytorch implementation of [Neural Sequence Decoder](https://github.com/fwillett/speechBCI/tree/main/NeuralDecoder)

## Requirements
- python >= 3.9

## Installation

pip install -e .

## How to run

1. Convert the speech BCI dataset using [formatCompetitionData.ipynb](./notebooks/formatCompetitionData.ipynb)
2. Train model: `python ./scripts/train_model.py`
