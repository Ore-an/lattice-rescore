#!/bin/bash
set -e

cmd=run.pl

nj=${1}
dataset=${2}
ngram=${3}
isca_1=${4}
spm=${5}

vis_devices=(3 4 5 6)

# activate virtualenv
. ./path.sh

# split json file
if [ ! -d "./dump/${dataset}/deltafalse/split${nj}utt" ]; then
    splitjson.py \
        --parts ${nj} \
        ./dump/${dataset}/deltafalse/data_unigram5000.json
fi

# submit jobs
latdir="./tdnn_1d_sp_htk/decode_${dataset}_fisher_tgpr_b8.0_d75_nolm"
latdir_out=${latdir}_rescore_espnet/
mkdir -p ${latdir_out}/log
# echo $0 $@ >> ${latdir_out}/log/CMD
# ${cmd} JOB=1:${nj} ${latdir_out}/log/get_lattice.JOB.log \
#     utils/distribute_gpus.sh "${vis_devices[@]}" -- JOB\
#     python ./get_lattice.py \5C
#     ${latdir} \
#     ${latdir_out} \
#     ${ngram} \
#     --isca_path ${isca_1} \
#     --spm_path ${spm} \
#     --js_path ./dump/${dataset}/deltafalse/split${nj}utt/data_unigram5000.JOB.json \
#     --gpu 

${cmd} JOB=1:${nj} ${latdir_out}/log/lattice_to_fst.JOB.log \
       python utils/htk_lat_to_fst.py --fudge-factor 5.0 --acoustic-wt --lm\
       ${latdir_out} \
       ${latdir_out}/words.txt JOB
