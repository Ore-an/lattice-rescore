import sys
import os
import tqdm
import gzip
import glob
import re
import argparse
from pathlib import Path
import openfst_python as fst


def make_fst(slf, syms, acwt=False, lm=False):
    lattice_info = {'nodes': {}, 'arcs': {}}
    for line in slf.decode().split('\n'):
        if line:
            if line[0] == 'N':
                m = re.match('N=(\d+)\s+L=(\d+)', line)
                lattice_info['nodes_n'] = int(m.groups()[0])
                lattice_info['arcs_n'] = int(m.groups()[1])
            elif line[0] == 'I':
                m = re.match('I=(\d+)\s+t=(\d+\.\d+)\s+W=(!NULL||<.?s>|[A-Za-z\'\\\\]+)\s+v=(\d+)', line)
                node_i = int(m.groups()[0])
                lattice_info['nodes'][node_i] = {'t': float(m.groups()[1]), 'label': m.groups()[2],
                                                 'v': int(m.groups()[3])}
            elif line[0] == 'J':
                m = re.match('J=(\d+)\s+S=(\d+)\s+E=(\d+)\s+a=(-?\d+.\d+)\s+l=(-?\d+.\d+)\s+i=(-?\d+.\d+)', line)
                arc_i = int(m.groups()[0])
                lattice_info['arcs'][arc_i] = {'start': int(m.groups()[1]), 'end': int(m.groups()[2]),
                                               'a_wt': float(m.groups()[3]), 'l_wt': float(m.groups()[4]),
                                               'i_wt': float(m.groups()[5])}

    if not lattice_info.get('nodes_n'):
        return None
    f_htk = fst.Fst()

    for i in range(lattice_info['nodes_n']):
        f_htk.add_state()
    for i in range(lattice_info['arcs_n']):
        info = lattice_info['arcs'][i]
        label = lattice_info['nodes'][info['end']]['label']
        if label == '<s>' or label == '</s>':
            label = '<eps>'
        label = syms.find(label)
        prob = -info['i_wt']  # kaldi uses NLL, it seems the rescoring is LL 
        if acwt:
            prob += info['a_wt']
        if lm:
            prob += info['l_wt']
        f_htk.add_arc(info['start'], fst.Arc(label, label, prob, info['end'])) 
    f_htk.set_start(0)
    end = [x for x in lattice_info['nodes'].keys() if lattice_info['nodes'][x]['label'] == '</s>']
    for e in end:
        f_htk.set_final(e)
    return f_htk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add i field from one lattice to another.")
    parser.add_argument("lat_dir", type=str, help="Directory containing the htk lattices")
    parser.add_argument("lat_dir_i", type=str, help="Directory containing the htk lattices with i scores")
    parser.add_argument("out_dir",  type=str, help="Output directory.")
    parser.add_argument("job", type=int, help="Number of this job in the run.pl. Opens the subdirectory of the lat dir.")
    args = parser.parse_args()

    outdir = args.out_dir
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for fn in tqdm.tqdm(glob.glob('{}/{}/*.gz'.format(args.lat_dir, args.job))):
        uttid = Path(fn).stem
        with gzip.open(fn, 'rb') as f:
            slf = f.read()
            conv_fst = make_fst(slf, syms, args.acoustic_wt, args.lm)
        if conv_fst:
            conv_fst = fst.determinize(conv_fst.rmepsilon()).minimize()
            out_ark.write('{} \n'.format(uttid[:-4]))
            out_ark.write(str(conv_fst))
            out_ark.write('\n')
        else:
            print()
            print('Error on lattice {}'.format(fn))
