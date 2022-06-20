#!/bin/bash

echo $0 $@

stage=1
cmd=run.pl
tscale=10.0 # the transition probability scale
original_scores=false
lm_fst_format=fst
remove_lm=false
model=

. ./path.sh
. parse_options.sh

set -euo pipefail

if [ $# -ne 3 ]; then
  echo "Convert Kaldi lattices to word level. Options to remove LM scores and push transition probabilities to acoustic scores."
  echo "Usage: $0 [options] <lang> <src-lattice-dir> <tgt-lattice-dir>"
  echo "e.g.:  "
  echo ""
  echo "Options:"
  echo "--cmd              (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "--original-scores  If true, do not modify LM and AC scores. If false, remove LM scores and push transition probabilities to AC score (default: $original_scores)"
  echo "--tscale           Transition probability scale. Only applicable if --original-scores is false (default: $tscale)"
  echo "--lm-fst-format    Format of language model FST in <lang>/G.* . Either fst or carpa (default: $lm_fst_format)"
  exit 1;
fi

lang=$1
src_dir=$2
dir=$3

if [ -z "$model" ]; then
  model=$src_dir/../final.mdl # assume model one level up from decoding dir.
fi

for x in $src_dir/lat.1.gz $model; do
  if [ ! -f $x ]; then
    echo "Error: missing file $x"
    exit 1
  fi
done

if [ "$original_scores" = "false" ]; then
  if [ "$lm_fst_format" = "fst" ]; then
    [ ! -f "$lang/G.fst" ] && echo "Error: missing file $lang/G.fst" && exit 1
  elif [ "$lm_fst_format" = "carpa" ]; then
    [ ! -f "$lang/G.carpa" ] && echo "Error: missing file $lang/G.carpa" && exit 1
  else
    echo "Error: unrecognised --lm-fst-format $lm_fst_format" && exit 1
  fi
fi

nj=`cat $src_dir/num_jobs`
mkdir -p $dir/log
echo $0 $@ >> $dir/CMD

if [ $stage -le 1 ]; then
  # remove LM score, fix timings, determinize, convert to FST
  # need to remove LM score, so that transition probability can be isolated and pushed to AC score, following the convention in HTK
  # need to align words to fix timing information
  if [ ! -e "$dir/fst.scp" ]; then
    if [ "$original_scores" = "false" ]; then
      if [ "$remove_lm" = "true" ]; then
        # remove LM score, merge transition probabilities into AC score
        if [ "$lm_fst_format" = "fst" ]; then
          $cmd JOB=1:$nj $dir/log/process_lattice.JOB.log \
            lattice-lmrescore --lm-scale=-1.0 "ark:gunzip -c $src_dir/lat.JOB.gz |" "fstproject --project_output=true $lang/G.fst |" ark:- \| \
            lattice-align-words --output-error-lats=true $lang/phones/word_boundary.int $model ark:- ark,t:- \| \
            utils/int2sym.pl -f 3 $lang/words.txt | gzip -c > $dir/JOB.lat.gz || exit 1
        elif [ "$lm_fst_format" = "carpa" ]; then
          $cmd JOB=1:$nj $dir/log/process_lattice.JOB.log \
            lattice-lmrescore-const-arpa --lm-scale=-1.0 "ark:gunzip -c $src_dir/lat.JOB.gz |" "$lang/G.carpa" ark:- \| \
            lattice-align-words --output-error-lats=true $lang/phones/word_boundary.int $model ark:- ark,t:- \| \
            utils/int2sym.pl -f 3 $lang/words.txt | gzip -c > $dir/JOB.lat.gz || exit 1
        else
          echo "Error: unrecognised --lm-fst-format $lm_fst_format"
          exit 1
        fi
      else
        # retain LM score, merge transition probabilities into AC score
        if [ "$lm_fst_format" = "fst" ]; then
          $cmd JOB=1:$nj $dir/log/process_lattice.JOB.log \
            lattice-lmrescore --lm-scale=-1.0 "ark:gunzip -c $src_dir/lat.JOB.gz |" "fstproject --project_output=true $lang/G.fst |" ark:- \| \
            lattice-scale --lm2acoustic-scale=$tscale --lm-scale=0.0 ark:- ark:- \| \
            lattice-lmrescore --lm-scale=1.0 ark:- "fstproject --project_output=true $lang/G.fst |" ark:- \| \
            lattice-align-words --output-error-lats=true $lang/phones/word_boundary.int $model ark:- ark,t:- \| \
            utils/int2sym.pl -f 3 $lang/words.txt | gzip -c > $dir/JOB.lat.gz || exit 1
        elif [ "$lm_fst_format" = "carpa" ]; then
          $cmd JOB=1:$nj $dir/log/process_lattice.JOB.log \
            lattice-lmrescore-const-arpa --lm-scale=-1.0 "ark:gunzip -c $src_dir/lat.JOB.gz |" "$lang/G.carpa" ark:- \| \
            lattice-scale --lm2acoustic-scale=$tscale --lm-scale=0.0 ark:- ark:- \| \
            lattice-lmrescore-const-arpa --lm-scale=1.0 ark:- "$lang/G.carpa" ark:- \| \
            lattice-align-words --output-error-lats=true $lang/phones/word_boundary.int $model ark:- ark,t:- \| \
            utils/int2sym.pl -f 3 $lang/words.txt | gzip -c > $dir/JOB.lat.gz || exit 1
        else
          echo "Error: unrecognised --lm-fst-format $lm_fst_format"
          exit 1
        fi
      fi
    else
      # retain (LM+transition prob) and AC scores as they are in Kaldi lattice
      $cmd JOB=1:$nj $dir/log/process_lattice.JOB.log \
        lattice-align-words --output-error-lats=true $lang/phones/word_boundary.int $model "ark:gunzip -c $src_dir/lat.JOB.gz |" ark,t:- \| \
        utils/int2sym.pl -f 3 $lang/words.txt | gzip -c > $dir/JOB.lat.gz || exit 1
    fi

    # make list of lattices
    find -L $PWD/$dir/ -name '*.lat.gz' > $dir/fst.scp || exit 1
  fi
fi

echo "Done!"

