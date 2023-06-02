# Counterfactual World Models
An approach to building pure vision foundation models by prompting masked predictors with "counterfactual" visual inputs.

## Setup
We recommend installing required packages in a virtual environment, e.g. with venv or conda.

1. git clone git@github.com:neuroailab/CounterfactualWorldModels.git
2. cd CounterfactualWorldModels && pip install -e .

### Pretrained Models
Weights are currently available for two VMAEs trained with the _temporall-factored masking policy_:
- A ViT-base VMAE with 8x8 patches, trained 3200 epochs on Kinetics400
- A ViT-large VMAE with 4x4 patches, trained 100 epochs on Kinetics700 + Moments + (20% of Ego4D)

See demo jupyter notebooks for urls to download these weights and load them into VMAEs.

## Run a demo of making factual predictions

Run the jupyter notebook `CounterfactualWorldModels/demo/FactualAndCounterfactual.ipynb`

## Coming Soon!
- [ ] Run counterfactual simulations (a.k.a. predictions) on user-supplied images
- [ ] Head-motion (i.e. IMU)-conditioned prediction models
- [ ] Clicking-based Spelke object segmentation interface
