#!/bin/bash

stage=1
cmd='run.pl'
isca=./espnet/train_960_pytorch_base/results/model.val5.avg.best
spm=./espnet/spm/lang_char/train_960_unigram5000.model
js_path=./dump/ihm_train/deltafalse/data_unigram5000.json
format='kaldi'
suffix='.gz'

. ./path.sh
. parse_options.sh

indir=$1
odir=$2
ngram=$3

set -x

nj=20

$cmd JOB=1:$nj ${odir}/log/rescore_lattice.JOB.log \
     python get_lattice.py \
     --isca_path $isca \
     --spm_path $spm \
     --js_path $js_path \
     --format $format \
     --suffix .JOB${suffix} \
     $indir $odir $ngram
