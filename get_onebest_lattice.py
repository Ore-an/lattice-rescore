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
from tqdm.contrib.concurrent import process_map, thread_map
from functools import partial
from collections import defaultdict

import lattice
from lattice import Lattice
import lattice_train
import utils

def lattice_expand(from_iterator, ngram=3, lat_format='htk', acwt=None, lmwt=None):
    """Lattice expansion and compute RNNLM and/or ISCA scores."""
    uttid, lat_in, lat_out, feat_path = from_iterator
    try:
        lat = Lattice(lat_in, file_type=lat_format)
    except FileNotFoundError:
        return (uttid, {})
    results = {}
    for aw in acwt:
        results_aw = {}
        for lw in lmwt:
            if abs(lw) != abs(aw) or abs(aw) == 1.0 or abs(lw) == 1.0:
                hyp = [arc.dest.sym for arc in lat.onebest(aw=aw, lw=lw)]
                # sos, eos, and other tokens are stripped
                hyp = [x for x in hyp if x not in lattice.SPECIAL_KEYS]
                hyp = ' '.join(hyp)
                results_aw[lw] = hyp
        results[aw] = results_aw
    return (uttid, results)


def main():
    parser = argparse.ArgumentParser(
        description='Lattice expansion and rescoring. '
                    'This should be run on the queue.')
    parser.add_argument('indir', type=str,
                        help='Input lattice directory.')
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
    parser.add_argument('--acwt', type=float, default=[1.0], nargs='+',
                        help='Acoustic scaling factor.')
    parser.add_argument('--lmwt', type=float, default=[1.0], nargs='+',
                        help='LM scaling factor.')
    args = parser.parse_args()

    # fh = logging.FileHandler('e2e_on_lattice_results_{}/log'.format(args.tag))
    # fh.setLevel(logging.DEBUG)
    # logger.addHandler(fh)

    # read acronym mapping
    if args.acronyms:
        acronyms = utils.load_acronyms(args.acronyms)
    else:
        acronyms = {}

    # set up RNNLM
    if args.rnnlm_path:
        rnnlms = []
        for rnnlm_path in args.rnnlm_path:
            rnnlm = utils.load_espnet_rnnlm(rnnlm_path)
            rnnlms.append(rnnlm)
    else:
        rnnlms = None

    # set up ISCA
    if args.isca_path:
        from espnet.utils.io_utils import LoadInputsAndTargets
        iscas, loaders = [], []
        for isca_path in args.isca_path:
            if args.gpu:
                device = 'cuda'
            else:
                device = 'cpu'
            model, char_dict, train_args = utils.load_espnet_model(isca_path, device=device)
            module_name = train_args.model_module
            # for param in model.encoder.parameters():
            #     param.requires_grad = False
            # model.eval()
            if 'transformer' in module_name or 'conformer' in module_name:
                model_type = 'tfm'
            else:
                model_type = 'las'
            loader = LoadInputsAndTargets(
                mode='asr',
                load_output=False,
                sort_in_input_length=False,
                preprocess_conf=train_args.preprocess_conf,
                preprocess_args={'train': False},
            )
            iscas.append((model, char_dict, model_type))
            loaders.append(loader)
            optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    else:
        iscas, loaders, js, optim = None, None, None, None

    # get sentencepiece model if needed
    if args.spm_path:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(args.spm_path)
    else:
        sp = None


    all_time = None
    counter = 0
    # set up iterator and run all

    # js_files = glob.glob(args.js_path)
    # js = {}
    # for js_path in js_files:
    #     with open(js_path, 'rb') as fh:
    #         js.update(json.load(fh)['utts'])
    acwt = args.acwt
    lmwt = args.lmwt
    with open(args.indir) as f:
        tot_files = len(f.readlines())
    all_lat = utils.list_iterator(args.indir, '.lat.gz')
    tot_loss = 0.
    batch_loss = 0.
    funcpart = partial(lattice_expand, acwt=acwt, lmwt=lmwt)
    j = process_map(funcpart, all_lat, total=tot_files, max_workers=50)

    files_text = defaultdict(list)
    try:
        for utt in sorted(j, key=lambda x: x[0]):
            uttid, results = utt
            for aw in acwt:
                for lw in lmwt:
                    if abs(lw) != abs(aw) or abs(aw) == 1.0:
                        files_text[f"pyth_ob_acwt{aw}_lmwt{lw}"].append(f"{uttid} {results[aw][lw]}\n")
        for k in files_text.keys():
            with open(k, 'w') as f:
                for line in files_text[k]:
                    f.write(line)
    except:
        from IPython import embed
        embed()

if __name__ == '__main__':
    main()
