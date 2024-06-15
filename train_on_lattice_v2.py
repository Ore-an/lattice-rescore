#!/usr/bin/env python3

import os
import sys
import json
import argparse
import logging
import time
import atexit

import numpy as np
import torch
import glob
import tqdm
import random
random.seed(10)

import pickle
import multiprocessing

from tqdm.contrib.concurrent import process_map, thread_map
from copy import deepcopy
from functools import partial
from itertools import repeat

import lattice
from lattice import Lattice
import lattice_train_v2 as lattice_train
import utils


def lattice_expand(from_iterator, ngram=3, gsf=None, rnnlms=None, isca=None,
                   sp=None, loader=None, overwrite=False, acronyms={}, lat_format='htk',
                   gpu=False, optim=None, onebest=False, max_arcs=750, soft_ob=False):
    """Lattice expansion and compute RNNLM and/or ISCA scores."""
    uttid, lat, lat_out, feat_path = from_iterator
    # logging.info('Processing lattice %s' % uttid)
    if gsf is None:
        gsf = float(lat.header['lmscale'])
    assert isca is not None, "model is needed for training"
    assert loader is not None, 'loader is needed for training'
    assert sp is not None, 'sp model is needed for training'
    logging.debug('Lattice arcs: %d' % lat.num_arcs())
    model, char_dict, model_type = isca
    feat = loader([(uttid, feat_path)])[0][0]
    if gpu:
        device = 'cuda'
    else:
        device = 'cpu'
        # Run forward-backward on lattice
    loss = lattice_train.isca_rescore(
            lat,
            feat,
            model,
            char_dict,
            ngram,
            sp,
            acronyms=acronyms,
            device=device,
            optim=optim,
            max_arcs=max_arcs,
            onebest=onebest,
            soft_ob=soft_ob
        )
    # logging.info('Lattice loss: %f' % loss)
    logging.debug('Processed lattice %s' % uttid)
    return loss

def load_lattice(lat_info, onebest=False, acwt=1.0, lmwt=1.0, ff=1.0, lat_format='htk', queue=None):
    uttid, lat_in, lat_out, feat_path = lat_info
    try:
        lat = Lattice(lat_in, file_type=lat_format)
        lat.fix_scores(aw=acwt/ff, lw=lmwt/ff)
        if onebest:
            lat = lat.onebest_lat()
        lat.posterior()
        out = uttid, lat, lat_out, feat_path
        queue.put(out)
        return 0
    except (FileNotFoundError, KeyError) as error:
        logging.info(f"Error {error} on {uttid}")
        return 1


class PickleLoader:
    def __init__(self, fn, discard_fn='', min_arcs=2):
        self.pkl = open(fn, 'rb')
        self.pkl_idx = {}
        self.discard = {}
        self.min_arcs = min_arcs
        self.load_discard(discard_fn)
        self.load_idxs(fn)
        self.utt_order = []
        self.shuffle()

    def load_discard(self, discard_fn):
        if discard_fn and os.path.isfile(discard_fn):
            with open(discard_fn) as f:
                for line in f:
                    line_spl = line.split()
                    self.discard[line_spl[0]] = (line_spl[1], line_spl[2])

    def load_idxs(self, fn):
        with open(fn+'.idx', 'r') as f:
            for line in f:
                line_spl = line.split()
                if line_spl[0] not in self.discard:
                    self.pkl_idx[line_spl[0]] = int(line_spl[1])

    def __len__(self):
        return len(self.pkl_idx)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __del__(self):
        self.pkl.close()
        del(self.pkl_idx)
        del(self.utt_order)

    def get_info(self, uttid):
        self.pkl.seek(self.pkl_idx[uttid])
        return pickle.load(self.pkl)

    def next(self):
        if len(self.utt_order) > 0:
            for i, uttid in enumerate(self.utt_order):
                self.utt_order = self.utt_order[i + 1:]
                info = self.get_info(uttid)
                if info and info[1].num_arcs() > self.min_arcs:
                    return info
                else:
                    self.pkl_idx.pop(uttid)
                    self.discard[uttid] = (info[1].num_arcs(), info[1].num_nodes())
        else:
            raise StopIteration()

    def shuffle(self):
        self.utt_order = list(self.pkl_idx.keys())
        random.shuffle(self.utt_order)

def pickle_write(fn, queue):
    with open(fn, 'wb') as f, open(fn + '.idx', 'w') as fi:
        while True:
            lat = queue.get()
            if lat == "stop":
                break
            idx = f.tell()
            f.write(pickle.dumps(lat))
            f.flush()
            fi.write(f'{lat[0]} {idx}\n')
            fi.flush()

def fix_end_node(info):
    uttid, lat, lat_out, feats = info
    lat.max_arc_idx = None
    lat.end[1].entries = [x for x in lat.end[1].entries if x in lat.arcs.keys()]
    return uttid, lat, lat_out, feats

def freeze_params(model, prefixes, logger):
    for mod, param in model.named_parameters():
        if any(mod.startswith(m) for m in prefixes):
            logger.warning(f"Freezing {mod}. It will not be updated during training.")
            param.requires_grad = False
    model_params = filter(lambda x: x.requires_grad, model.parameters())
    return model, model_params

def main():
    parser = argparse.ArgumentParser(
        description='Lattice expansion and rescoring. '
                    'This should be run on the queue.')
    parser.add_argument('indir', type=str,
                        help='Input lattice directory.')
    parser.add_argument('ngram', type=int,
                        help='Ngram expansion approximation.')
    parser.add_argument('--rnnlm-path', type=str, default=None,
                        help='Path to rnnlm model.')
    parser.add_argument('--isca-path', type=str, default="./espnet/train_960_pytorch_base/results/model.val5.avg.best",
                        help='Path to isca model.')
    parser.add_argument('--spm-path', type=str, default="./espnet/spm/lang_char/train_960_unigram5000.model",
                        help='Path to sentencepiece model.')
    parser.add_argument('--js-path', type=str, default="./dump/ihm_train/deltafalse/split12utt/data_unigram5000.*.json",
                        help='Path to json feature file for ISCA.')
    parser.add_argument('--gsf', type=float, default=1.0,
                        help='Grammar scaling factor.')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite existing output file if it exists.')
    parser.add_argument('--acronyms', type=str, default=None,
                        help='Path to acronoym mapping (swbd)')
    parser.add_argument('--format', type=str, default='kaldi', choices=['htk', 'kaldi'], dest='lat_format',
                       help='Format of the lattices.')
    parser.add_argument('--suffix', type=str, default='.gz',
                        help='Suffix of the lattices.')
    parser.add_argument('--ff', type=int, default=1,
                        help='Scaling fudge factor for scores on lattice')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for espnet model')
    parser.add_argument('--fix', action='store_true',
                        help='Fix bugged lattices')
    parser.add_argument('--onebest', action='store_true',
                        help='Train on best path only')
    parser.add_argument('--soft-ob', action='store_true',
                        help='Loss is hyb score - e2e score even in ob.')
    parser.add_argument('--max-arcs', type=int, default=750,
                        help="Max number of arcs backprop fitting on GPU.")
    parser.add_argument('--tag', type=str, default='base',
                        help="Tag to add to exp dir.")
    parser.add_argument('--acwt', type=float, default=1.0,
                        help='Acoustic scaling factor.')
    parser.add_argument('--lmwt', type=float, default=1.0,
                        help='LM scaling factor.')
    parser.add_argument('--eps', type=int, default=4,
                        help='Number of epochs.')
    parser.add_argument('--subset', type=int,
                        help='Number of utterances to train on.')
    parser.add_argument('--min-arcs', type=int, default=2,
                        help='Discard lattices with less than n arcs.')
    parser.add_argument('--accum-steps', type=int, default=32,
                        help="Number of lattices to accumulate gradients for before updating model.")
    parser.add_argument("--freeze-params",
                        type=str,
                        default=[],
                        nargs="*",
                        help="Freeze parameters")
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()


    outdir = "e2e_on_lattice_results_{}".format(args.tag)
    if args.ff != 1:
        outdir = outdir + "_trainff{}".format(args.ff)
    outdir = outdir + "_acwt{}_lmwt{}_v2".format(args.acwt, args.lmwt)
    os.makedirs(outdir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename='{}/log'.format(outdir),
        format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
    )
    logger = logging.getLogger()

    # fh = logging.FileHandler('e2e_on_lattice_results_{}/log'.format(args.tag))
    # fh.setLevel(logging.DEBUG)
    # logger.addHandler(fh)

    logger.info(' '.join([sys.executable] + sys.argv))

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
        if args.gpu:
            device = 'cuda'
        else:
            device = 'cpu'
        model, char_dict, train_args = utils.load_espnet_model(args.isca_path, device=device, mode='train')
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
            preprocess_args={'train': True},
        )
        isca = (model, char_dict, model_type)
        if args.freeze_params:
            model, model_params = freeze_params(model, args.freeze_params, logger)
        else:
            model_params = model.parameters()
        optim = torch.optim.Adam(model_params, lr=1e-4)
    else:
        isca, loader, js, optim = None, None, None, None

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

    js_files = glob.glob(args.js_path)
    js = {}
    for js_path in js_files:
        with open(js_path, 'rb') as fh:
            js.update(json.load(fh)['utts'])
    if not os.path.exists(f'{outdir}/ds.1.pkl') and not os.path.exists(f'{outdir}/ds.pkl'):
        mgr = multiprocessing.Manager()
        lat_q = mgr.Queue()
        pool = multiprocessing.Pool(16)
        pickler = pool.apply_async(pickle_write, (f'{outdir}/ds.pkl', lat_q))
        if args.lat_format == 'kaldi':
            all_lat = list(utils.kaldiLatticeIterator(args.indir, args.suffix, resource=js))
        else:
            all_lat = list(utils.list_iterator(args.indir, '.lat.gz', resource=js))
        if args.subset:
            all_lat = all_lat[:args.subset]
        load_lat_partial = partial(load_lattice, onebest=args.onebest, acwt=args.acwt, lmwt=args.lmwt, lat_format=args.lat_format, queue=lat_q, ff=args.ff)
        pbar = tqdm.tqdm(total=len(all_lat))
        def update(*a):
            pbar.update()
        jobs = []
        
        for ln in all_lat:
            j = pool.apply_async(load_lat_partial, (ln,), callback=update)
            jobs.append(j)
        for j in jobs:
            j.get()

        lat_q.put("stop")
        pool.close()
        pool.join()
    
    tot_eps = args.eps
    loading_lats = PickleLoader(f'{outdir}/ds.pkl', f'{outdir}/discard', args.min_arcs)

    if args.lat_format != 'kaldi':
        with open(args.indir) as f:
            tot_files = len(f.readlines())
    else:
        tot_files = len(loading_lats)
    
    tot_loss = 0.
    batch_loss = 0.
    loaded_lats = {}
    eps_done = 0
    maxarcs = 0
    maxnodes = 0
    ep_loss = 0.
    ep_counter = 0.
    if args.resume:
        ckpts = glob.glob('{}/ckpt.*.pt'.format(outdir))
        ckpt = ''
        mtime = 0
        for ckpf in ckpts:
            f_mtime = os.path.getmtime(ckpf)
            if f_mtime > mtime:
                mtime = f_mtime
                ckpt = ckpf
        if ckpt:
            ckpt = torch.load('{}/ckpt.tmp.pt'.format(outdir))
            model.load_state_dict(ckpt['model_state_dict'])
            optim.load_state_dict(ckpt['optimizer_state_dict'])
            counter = ckpt['counter']
            ep_counter = ckpt['ep_counter']
            tot_loss = ckpt['tot_loss']
            batch_loss = ckpt['batch_loss']
            ep_loss = ckpt['ep_loss']
            all_time = ckpt['all_time']
            loading_lats.utt_order = ckpt['utts_left']
            eps_done = ckpt['ep']
            print(f"Loaded {ckpts[0]}")
        else:
            print("No checkpoint to resume")
    while eps_done < tot_eps:
        for each_iteration in tqdm.tqdm(loading_lats, total=tot_files):
            arcs = each_iteration[1].num_arcs()
            nodes = each_iteration[1].num_nodes()
            try:
                loss = lattice_expand(
                    each_iteration, args.ngram, args.gsf, rnnlms, isca, sp, loader,
                    args.overwrite, acronyms, lat_format=args.lat_format, gpu=args.gpu,
                    optim=optim, onebest=args.onebest, max_arcs=args.max_arcs, soft_ob=args.soft_ob)
                loss = loss/args.accum_steps
                loss.backward()
                loss_num = loss.item()

            except Exception as e:
                print(f"Max arcs/arcs: {maxarcs}/{arcs} Max nodes/nodes:{maxnodes}/{nodes}")
                with open(f'{outdir}/discard', 'w') as f:
                    for k,v in loading_lats.discard.items():
                        f.write(f'{k} {v[0]} {v[1]}\n')
                    f.write(f'{each_iteration[0]} {arcs} {nodes}\n')
                torch.save({
                    'counter': counter,
                    'ep_counter': ep_counter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'tot_loss': tot_loss,
                    'batch_loss': batch_loss,
                    'ep_loss': ep_loss,
                    'all_time': all_time,
                    'utts_left': loading_lats.utt_order,
                    'ep': eps_done,
                }, '{}/ckpt.tmp.pt'.format(outdir))
                raise e

            if (counter + 1) % args.accum_steps == 0:
                optim.step()
                optim.zero_grad()
            if arcs > maxarcs:
                maxarcs = arcs
            if nodes > maxnodes:
                maxnodes = nodes
            tot_loss += loss_num
            batch_loss += loss_num
            ep_loss += loss_num
            counter += 1
            ep_counter += 1

            # if counter % 100 == 0:
            #     reporter.report(verbose=True)
            if counter % 1000 == 0:
                logger.info("Avg loss: {:.2f} Last 1000 utt: {:.2f} Ep loss: {:.2f} Done: {}".format((tot_loss/counter), (batch_loss/1000), (ep_loss/ep_counter), counter))
                batch_loss = 0.
        eps_done += 1
        logger.info("Ep {} loss: {:.2f}".format(eps_done, (ep_loss/ep_counter)))
        torch.save({
            'it': counter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': tot_loss/counter,
            'ep': eps_done,
        }, '{}/ckpt.ep{}.pt'.format(outdir, eps_done))
        ep_loss = 0.
        ep_counter = 0.
        loading_lats.shuffle()
        tot_files = len(loading_lats)

    torch.save(model.state_dict(), '{}/model.pt'.format(outdir))

    logger.info('Job finished on %s' % os.uname()[1])

if __name__ == '__main__':
    main()
