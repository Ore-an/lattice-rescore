#!/bin/bash

echo $0 $@

stage=1
cmd='run.pl'
tscale=1.0
kaldilm='data/lang_test_fisher_tgpr'
kaldilmformat='fst' # for swbd, the default kaldi lm is in arpa format

. ./path.sh
. parse_options.sh

set -euo pipefail

kaldi_lat_dir=$1
proc_lat_dir=$2
proc_lat_dir_nolm=${proc_lat_dir}_nolm
mkdir -p $proc_lat_dir

echo "=========================" >> $proc_lat_dir/../CMD
echo "#$PWD" >> $proc_lat_dir/../CMD
echo $0 $@ >> $proc_lat_dir/../CMD


if [ $stage -le 1 ]; then
    mkdir -p $proc_lat_dir_nolm
    echo "Processing Kaldi lattices in ${kaldi_lat_dir} and saving to ${proc_lat_dir_nolm}"
    utils/process_lattice.sh \
        --cmd "$cmd" \
        --tscale $tscale \
        --original-scores false \
        --remove-lm true \
        --lm-fst-format ${kaldilmformat} \
        $kaldilm \
        $kaldi_lat_dir \
        $proc_lat_dir_nolm
fi

echo "Done!"
