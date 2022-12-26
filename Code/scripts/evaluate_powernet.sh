#!/bin/sh
python train.py --model powernet --dataset riding_data --evaluate --exp-load-weights-from scripts/ckpt/powernet_qat_best3-q.pth.tar -8 --save-sample 0 --use-bias --device MAX78000 "$@"
