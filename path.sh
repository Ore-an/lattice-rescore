export CONDA_ENVS_PATH=/disk/scratch4/acarmant/software/miniconda3/envs
export CONDA_PKGS_DIRS=/disk/scratch4/acarmant/software/miniconda3/pkgs

export KALDI_ROOT=/disk/scratch4/acarmant/kaldi

export LD_LIBRARY_PATH=$KALDI_ROOT/src/lib:$KALDI_ROOT/tools/openfst-1.6.7/lib:$LD_LIBRARY_PATH
export PATH=$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/../kaldi_lm/:$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/kwsbin:$KALDI_ROOT/src/nnet3bin:$KALDI_ROOT/src/chainbin:$KALDI_ROOT/tools/sph2pipe_v2.5/:$KALDI_ROOT/src/rnnlmbin:$PWD:$PATH
export LC_ALL=C

# Directory under which the LM training corpus should be extracted
LM_CORPUS_ROOT=./lm-corpus
