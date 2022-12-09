echo $0 $@

stage=2
cmd='run.pl'
tscale=1.0
cfg='utils/hlrescore.cfg'
dict='data/local/dict/lexicon.txt'
kaldilm='/disk/scratch4/acarmant/kaldi/egs/librispeech/s5/data/lang_test_fisher_tgpr'
kaldilmformat='fst' # for swbd, the default kaldi lm is in arpa format
htklm='data/local/lm/fisher.o3g.kn'

. ./path.sh
. parse_options.sh

set -euo pipefail
HLRescore_BIN="/disk/scratch4/acarmant/software/htk/bin/HLRescore"
hlrescore="${HLRescore_BIN} -A -D -V -T 1"

kaldi_lat_dir=$1
htk_lat_dir=$2
htk_lat_dir_nolm=${htk_lat_dir}_nolm
mkdir -p $htk_lat_dir

echo "=========================" >> $htk_lat_dir/../CMD
echo "#$PWD" >> $htk_lat_dir/../CMD
echo $0 $@ >> $htk_lat_dir/../CMD


if [ $stage -le 1 ]; then
    mkdir -p $htk_lat_dir_nolm
    echo "Converting Kaldi lattices from ${kaldi_lat_dir} to HTK lattices in ${htk_lat_dir_nolm}"
    utils/convert_slf_parallel.sh \
        --cmd "$cmd" \
        --tscale $tscale \
        --original-scores false \
        --frame-rate 0.03 \
        --remove-lm true \
        --lm-fst-format ${kaldilmformat} \
        $kaldilm \
        $kaldi_lat_dir \
        $htk_lat_dir_nolm
fi

if [ $stage -le 2 ]; then
    nj=12
    rm -f ${htk_lat_dir}/lm/.error
    rm -f  ${htk_lat_dir}/lm/*/log 
    echo "Writing to ${htk_lat_dir}/lm"
    for i in $(seq 1 $nj); do
        (
        mkdir -p ${htk_lat_dir}/lm/$i
        for file in ${htk_lat_dir}/$i/*.lat.gz; do
	    ofn=${htk_lat_dir}/lm/${i}/$(basename $file)
	    if [ ! -f $ofn ]; then
            $hlrescore -C $cfg -n $htklm -w -l ${htk_lat_dir}/lm/$i \
                       $dict ${file%.gz} >>  ${htk_lat_dir}/lm/$i/log 2>&1 || echo "$file is broken" >> ${htk_lat_dir}/lm/$i/errors
	    fi
        done
        ) || touch $htk_lat_dir/.error &
    done
    wait
    if [ -f $htk_lat_dir/.error ]; then
        echo "$0: something went wrong for hlrescore"
        exit 1
    fi
fi

echo "Done!"
