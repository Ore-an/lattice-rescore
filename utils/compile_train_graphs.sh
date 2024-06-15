basedir=/disk/scratch4/acarmant/kaldi/egs/librispeech/s5/
lang_dir=${basedir}/data/lang_chain
gmm_dir=${basedir}/exp
fsts_dir=tdnn_1d_sp_htk/decode_ihm_train_fisher_tgpr_b8.0_d75_nolm_rescore_espnet/fsts
outdir=tdnn_1d_sp_htk/decode_ihm_train_fisher_tgpr_b8.0_d75_nolm_rescore_espnet/outfst

mkdir -p ${outdir}/log/
run.pl JOB=1:12 ${outdir}/log/compile_graphs.JOB.log compile-train-graphs-fsts --read-disambig-syms=${lang_dir}/phones/disambig.int \
                                                     --self-loop-scale=0.1 --transition-scale=1 \
						     ${gmm_dir}/tree \
						     ${gmm_dir}/final.mdl \
						     ${lang_dir}/L_disambig.fst \
						     ark,t:${fsts_dir}/fst.JOB.ark \
						     ark,scp:${outdir}/train_graph.JOB.ark,${outdir}/train_graph.JOB.scp
