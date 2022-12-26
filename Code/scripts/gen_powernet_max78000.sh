#!/bin/sh
DEVICE="MAX78000"
TARGET="sdk/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python ai8xize.py --test-dir $TARGET --prefix powernet --checkpoint-file trained/powernet_qat_best3-q.pth.tar --config-file networks/powernet.yaml --overwrite  $COMMON_ARGS "$@"
