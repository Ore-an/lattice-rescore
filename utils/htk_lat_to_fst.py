import sys
import os
import tqdm
import gzip
import glob
import re
import argparse
from pathlib import Path
import openfst_python as fst


def make_fst(slf, syms, acwt=False, lm=False, fudge_factor=1.0):
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
                if lm:
                    m = re.match('J=(\d+)\s+S=(\d+)\s+E=(\d+)\s+a=(-?\d+.\d+)\s+l=(-?\d+.\d+)\s+r=(-?\d+.\d+)', line)
                else:
                    m = re.match('J=(\d+)\s+S=(\d+)\s+E=(\d+)\s+a=(-?\d+.\d+)\s+l=(-?\d+.\d+)\s+i=(-?\d+.\d+)', line)
                arc_i = int(m.groups()[0])
                lattice_info['arcs'][arc_i] = {'start': int(m.groups()[1]), 'end': int(m.groups()[2]),
                                               'a_wt': float(m.groups()[3]), 'l_wt': float(m.groups()[4]),
                                               'i_wt': float(m.groups()[5])}

    if not lattice_info.get('nodes_n'):
        return None
    f_htk = fst.Fst('log')

    for i in range(lattice_info['nodes_n']):
        f_htk.add_state()
    for i in range(lattice_info['arcs_n']):
        info = lattice_info['arcs'][i]
        label = lattice_info['nodes'][info['end']]['label']
        if label == '<s>' or label == '</s>':
            label = '<eps>'
        label = syms.find(label)
        prob = -info['i_wt'] * fudge_factor # kaldi uses NLL, it seems the rescoring is LL 
        if acwt:
            prob += info['a_wt'] # acwt seems to still be NLL
        elif lm:
            prob = -info['l_wt']
        prob = fst.Weight('log', prob)
        f_htk.add_arc(info['start'], fst.Arc(label, label, prob, info['end'])) 
    f_htk.set_start(0)
    end = [x for x in lattice_info['nodes'].keys() if lattice_info['nodes'][x]['label'] == '</s>']
    for e in end:
        f_htk.set_final(e)
    return f_htk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform htk-style lattices into fsts.")
    parser.add_argument("lat_dir", type=str, help="Directory containing the htk lattices")
    parser.add_argument("syms_fn", type=str, help="Kaldi words.txt file")
    parser.add_argument("job", type=int, help="Number of this job in the run.pl. Opens the subdirectory of the lat dir.")
    parser.add_argument("--acoustic-wt", action="store_true", help="Flag to add acoustic weight from lf-mmi to espnet rescoring.")
    parser.add_argument("--lm", action="store_true", help="Flag to add lm weight from lf-mmi to espnet rescoring.")
    parser.add_argument("--fudge-factor", type=float, default=1.0, help="Weight to multiply the E2E score by")
    args = parser.parse_args()

    suff = ''
    if args.acoustic_wt:
        suff += "_ac"
    if args.lm:
        suff += "_lm"
    else:
        lm_fst = None
    if args.fudge_factor != 1.0:
        suff += "_ff{}".format(args.fudge_factor)
    syms = fst.SymbolTable.read_text(args.syms_fn)
    outdir = "{}/fsts{}/".format(args.lat_dir, suff)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    out_ark = open("{}/fsts{}/fst.{}.ark".format(args.lat_dir, suff, args.job), 'w')
    for fn in tqdm.tqdm(glob.glob('{}/{}/*.gz'.format(args.lat_dir, args.job))):
        uttid = Path(fn).stem
        with gzip.open(fn, 'rb') as f:
            slf = f.read()
            conv_fst = make_fst(slf, syms, args.acoustic_wt, fudge_factor=args.fudge_factor)
        if args.lm:
            try:
                with gzip.open("{}/lm/{}/{}.gz".format(args.lat_dir, args.job, uttid)) as f:
                    slf = f.read()
                    lm_fst = make_fst(slf, syms, False, True)
            except FileNotFoundError:
                conv_fst = None
        if conv_fst:
            conv_fst = fst.determinize(conv_fst.rmepsilon()).minimize()
            if lm_fst:
                lm_fst = fst.determinize(lm_fst.rmepsilon()).minimize()
                conv_fst = fst.intersect(conv_fst.arcsort(sort_type="olabel"), lm_fst)
            out_ark.write('{} \n'.format(uttid[:-4]))
            out_ark.write(str(conv_fst))
            out_ark.write('\n')
        else:
            print()
            print('Error on lattice {}'.format(fn))
    out_ark.close()
