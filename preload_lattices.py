#!/usr/bin/env python3

import os
import sys
import json
import argparse
import logging
import time
import numpy as np
import torch
import glob
import tqdm
import random
from tqdm.contrib.concurrent import process_map, thread_map
from copy import deepcopy
from functools import partial
from itertools import repeat

import multiprocessing
import pickle
import lattice
from lattice import Lattice
import lattice_train
import utils

def load_lattice(lat_info, onebest=False, acwt=1.0, lmwt=1.0, lat_format='htk'):
    uttid, lat_in, lat_out, feat_path = lat_info
    try:
        lat = Lattice(lat_in, file_type=lat_format)
        if onebest:
            lat = lat.onebest_lat(aw=acwt, lw=lmwt)
        return (uttid, lat, lat_out, feat_path)
    except FileNotFoundError:
        pass        


def main():
    parser = argparse.ArgumentParser(
        description='Lattice expansion and rescoring. '
                    'This should be run on the queue.')
    parser.add_argument('indir', type=str,
                        help='Input lattice directory.')
    parser.add_argument('ngram', type=int,
                        help='Ngram expansion approximation.')
    parser.add_argument('--rnnlm-path', type=str, default=None, action='append',
                        help='Path to rnnlm model.')
    parser.add_argument('--isca-path', type=str, default=None, action='append',
                        help='Path to isca model.')
    parser.add_argument('--spm-path', type=str, default=None,
                        help='Path to sentencepiece model.')
    parser.add_argument('--js-path', type=str, default=None,
                        help='Path to json feature file for ISCA.')
    parser.add_argument('--gsf', type=float, default=1.0,
                        help='Grammar scaling factor.')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite existing output file if exits.')
    parser.add_argument('--acronyms', type=str, default=None,
                        help='Path to acronoym mapping (swbd)')
    parser.add_argument('--format', type=str, default='htk', choices=['htk', 'kaldi'], dest='lat_format',
                       help='Format of the lattices.')
    parser.add_argument('--suffix', type=str, default='.lat',
                        help='Suffix of the lattices.')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for espnet model')
    parser.add_argument('--onebest', action='store_true',
                        help='Train on best path only')
    parser.add_argument('--max-arcs', type=int, default=750,
                        help="Max number of arcs backprop fitting on GPU.")
    parser.add_argument('--tag', type=str, default='base',
                        help="Tag to add to exp dir.")
    parser.add_argument('--acwt', type=float, default=1.0,
                        help='Acoustic scaling factor.')
    parser.add_argument('--lmwt', type=float, default=1.0,
                        help='LM scaling factor.')
    parser.add_argument('-j', '--job', type=str, default='base',
                        help="Job number.")
    args = parser.parse_args()


    outdir = "e2e_on_lattice_results_{}".format(args.tag)

    if args.onebest:
        outdir = outdir + "_acwt{}_lmwt{}".format(args.acwt, args.lmwt)
    os.makedirs(outdir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
    )
    logger = logging.getLogger()

    # fh = logging.FileHandler('e2e_on_lattice_results_{}/log'.format(args.tag))
    # fh.setLevel(logging.DEBUG)
    # logger.addHandler(fh)

    logging.info(' '.join([sys.executable] + sys.argv))

    # read acronym mapping
    if args.acronyms:
        acronyms = utils.load_acronyms(args.acronyms)
    else:
        acronyms = {}

    js_files = glob.glob(args.js_path + "*.json")
    js = {}
    for js_path in js_files:
        with open(js_path, 'rb') as fh:
            js.update(json.load(fh)['utts'])
    with open(args.indir) as f:
        tot_files = len(f.readlines())
    fn = f'{outdir}/ds.{args.job}.pkl'
    if not os.path.exists(fn):
        all_lat = list(utils.list_iterator(args.indir, '.lat.gz', resource=js))
        load_lat_partial = partial(load_lattice, onebest=args.onebest, acwt=args.acwt, lmwt=args.lmwt, lat_format=args.lat_format)
        loaded_lats = map(load_lat_partial, all_lat)
    with open(fn, 'wb') as f:
        for lat in loaded_lats:
            pickle.dump(lat, f)

if __name__ == '__main__':
    main()
