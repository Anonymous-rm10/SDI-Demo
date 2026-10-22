#!/bin/bash
# Local adaptation for the bundled layout; see THIRD_PARTY_NOTICES.md.

# Bundled layout: use the canonical package rather than upstream convenience links.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FIRERED_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$FIRERED_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$FIRERED_ROOT/fireredasr2s/fireredasr2:$FIRERED_ROOT/fireredasr2s/fireredasr2/utils:$FIRERED_ROOT/fireredasr2s/fireredvad/bin:$FIRERED_ROOT/fireredasr2s/fireredlid:$FIRERED_ROOT/fireredasr2s/fireredpunc:$PATH"

# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu, Yan Jia)

export PATH=$PWD/fireredlid/:$PWD/fireredlid/utils/:$PATH
export PYTHONPATH=$PWD/:$PYTHONPATH

if [ ! -f wav/wav.scp ]; then
    (cd wav && tar -zxvf wav.tar.gz)
fi

# model_dir includes model.pth.tar, cmvn.ark, dict.txt
model_dir=$PWD/pretrained_models/FireRedLID

# Support several input format
wavs="--wav_path wav/BAC009S0764W0121.wav"
wavs="--wav_paths wav/BAC009S0764W0121.wav wav/IT0011W0001.wav wav/TEST_NET_Y0000000000_-KTKHdZ2fb8_S00000.wav wav/TEST_MEETING_T0000000001_S00000.wav"
wavs="--wav_scp wav/wav.scp"
wavs="--wav_dir wav/"

out="out/lang.txt"

decode_args="--use_gpu 1 --use_half 0 --sort_wav_by_dur 1 --batch_size 64"

mkdir -p $(dirname $out)
set -x


CUDA_VISIBLE_DEVICES=0 \
speech2lang.py --model_dir $model_dir $decode_args $wavs --output $out
