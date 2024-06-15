kdir=/disk/scratch4/acarmant/kaldi/egs/librispeech/s5/
nj=12
for odir in tdnn_1d_sp/decode*; do
    utils/run.pl JOB=1:$nj ${odir}/log/lattice_depth.JOB.log \
        lattice-depth "ark:gunzip -c ${odir}/lat.JOB.gz| $kdir/utils/sym2int.pl -f 3 $kdir/data/lang_test_fisher_tgpr/words.txt|"  ark:/dev/null || exit 1;

    grep -w Overall ${odir}/log/lattice_depth.*.log | \
        awk -v nj=$nj '{num+=$6*$8; den+=$8; nl++} END{
          if (nl != nj) { print "Error: expected " nj " lines, got " nl | "cat 1>&2"; }
          printf("%.2f ( %d / %d )\n", num/den, num, den); }' > ${odir}/depth || exit 1;
    echo -n "$odir depth is: "
    cat ${odir}/depth
    done


