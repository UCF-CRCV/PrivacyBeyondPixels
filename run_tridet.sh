#!/bin/bash

python tridet/train.py tridet/thumos_videoMAE.yaml --output baseline
python tridet/eval.py tridet/thumos_videoMAE.yaml ckpt/thumos_videoMAE_baseline/epoch_039.pth.tar
