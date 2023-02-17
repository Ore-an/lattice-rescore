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

from lattice import Lattice
import lattice_train
import utils

def lattice_expand(from_iterator, ngram, gsf=None, rnnlms=None, iscas=None,
                   sp=None, loaders=None, overwrite=False, acronyms={}, lat_format='htk',
                   gpu=False, optim=None, onebest=False, max_arcs=750):
    """Lattice expansion and compute RNNLM and/or ISCA scores."""
    uttid, lat_in, lat_out, feat_path = from_iterator
    timing = []
    logging.info('Processing lattice %s' % uttid)
    start = time.time()
    lat = Lattice(lat_in, file_type=lat_format)
    if lat_format == 'kaldi':
        lat.dag2htk(out_lat + '.kaldi_test')
    if gsf is None:
        gsf = float(lat.header['lmscale'])
    if onebest:
        lat.onebest_lat(aw=-1)
    timing.append(time.time() - start)
    # if rnnlms is not None:
    #     for lm, word_dict in rnnlms:
    #         start = time.time()
    #         # Run forward-backward on lattice
    #         lat.posterior(aw=1/gsf)
    #         lat = lattice_train.rnnlm_rescore(lat, lm, word_dict, ngram)
    #         timing.append(time.time() - start)
    if iscas is not None:
        assert loaders is not None, 'loader is needed for ISCA rescoring'
        assert sp is not None, 'sp model is needed for ISCA rescoring'
        logging.debug('Lattice arcs: %d' % lat.num_arcs())
        for isca, loader in zip(iscas, loaders):
            model, char_dict, model_type = isca
            start = time.time()
            feat = loader([(uttid, feat_path)])[0][0]
            if gpu:
                device = 'cuda'
            else:
                device = 'cpu'
            # Run forward-backward on lattice
            lat.posterior(aw=1/gsf)
            loss = lattice_train.isca_rescore(
                lat,
                feat,
                model,
                char_dict,
                ngram,
                sp,
                model_type=model_type,
                acronyms=acronyms,
                device=device,
                optim=optim,
                max_arcs=max_arcs
            )
            timing.append(time.time() - start)
    logging.debug('Processed lattice %s' % uttid)
    logging.info('Lattice loss: %f' % loss)
    logging.debug('Time taken for %s: %s' % (
        uttid, ' '.join(['{:.3f}'.format(x) for x in timing])))
    return np.array(timing), loss

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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        filename='e2e_on_lattice_results_{}/log'.format(args.tag),
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
    else:
        iscas, loaders, js = None, None, None

    # get sentencepiece model if needed
    if args.spm_path:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(args.spm_path)
    else:
        sp = None

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    all_time = None
    counter = 0
    # set up iterator and run all

    js_files = glob.glob(args.js_path)
    js = {}
    for js_path in js_files:
        with open(js_path, 'rb') as fh:
            js.update(json.load(fh)['utts'])
    with open(args.indir) as f:
        tot_files = len(f.readlines())
    all_lat = utils.list_iterator(args.indir, '.lat.gz', resource=js)
    tot_loss = 0.
    batch_loss = 0.
    for each_iteration in tqdm.tqdm(all_lat,total=tot_files):
        timing, loss = lattice_expand(
            each_iteration, args.ngram, args.gsf, rnnlms, iscas, sp, loaders,
            args.overwrite, acronyms, lat_format=args.lat_format, gpu=args.gpu,
            optim=optim, onebest=args.onebest, max_arcs=args.max_arcs)
        tot_loss += loss
        batch_loss += loss
        if all_time is None:
            all_time = timing
        elif timing is not None:
            all_time += timing
        counter += 1
        if counter % 100 == 0:
            logging.info("Avg loss: {:.2f} Last 100 utt: {:.2f}".format((tot_loss/counter), (batch_loss/100)))
            batch_loss = 0.
        if counter % 10000 == 0:

            torch.save({
                'it': counter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': tot_loss/counter,
            }, 'e2e_on_lattice_results_{}/ckpt.{}.pt'.format(args.tag, counter))
    torch.save(model.state_dict(), 'e2e_on_lattice_results_{}/model.pt'.format(args.tag))

    logging.info('Job finished on %s' % os.uname()[1])
    logging.info('Overall, for %d lattices, %.3f seconds used'
                 % (counter, sum(all_time)))
    logging.info('On average, the time taken for each key part is %s'
                 % (' '.join(['{:.3f}'.format(x) for x in all_time / counter])))

if __name__ == '__main__':
    main()
