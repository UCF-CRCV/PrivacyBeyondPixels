#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
python mgfn/main.py --comment mae_baseline --fa_model mae --feature_size 768 --feat_extractor mae --batch_size 16