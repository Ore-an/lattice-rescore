#!/usr/bin/env python3

import math
from itertools import chain
from collections.abc import MutableMapping
from copy import deepcopy

import numpy as np
import torch

# from pytorch_memlab import profile_every
from espnet.nets.pytorch_backend.transformer.mask import target_mask
import lattice
from utils import sym2idx
from torch.profiler import profile, record_function
torch.autograd.set_detect_anomaly(True)
class Cache(dict):
    """A customised dictionary as the cache for lattice rescoring.
    Input list of string is first concatenated and then loop up.
    The key value pair is ngram: (state, pred, post).
    """
    __slots__ = 'model', 'dict', 'sp', 'h', 'cache_locality', 'acronyms', 'device', 'train', 'store_grad_cpu', 'sos', 'eos', 'zero', #'fwd_model'

    @staticmethod
    def _process_args(mapping=(), **kwargs):
        if hasattr(mapping, 'items'):
            mapping = getattr(mapping, 'items')()
        return ((k, v)
                for k, v in chain(mapping, getattr(kwargs, 'items')()))

    def __init__(self, model, dictionary, sp=None, cache_locality=9, acronyms={},
                 device='cpu', train=False, mapping=(), store_grad_cpu=False, **kwargs):
        super(Cache, self).__init__(self._process_args(mapping, **kwargs))
        self.model = model
        self.dict = dictionary
        self.sp = sp
        self.cache_locality = cache_locality
        self.acronyms = acronyms
        self.h = None
        self.device = device
        self.train = train
        self.store_grad_cpu = store_grad_cpu and (device != 'cpu')
        self.sos = torch.tensor([self.model.sos], dtype=torch.int64, device=self.device)
        self.eos = torch.tensor([self.model.eos], dtype=torch.int64, device=self.device)
        self.zero = torch.tensor(0.0, device=self.device)
        # self.fwd_model = None
        assert self.sp is not None, 'sentencepiece model required'
    
    def init_feat(self, feat):
        assert feat is not None, 'acoustic feature must be given'
        self.h = self.model.encode(torch.as_tensor(feat, device=self.device)).unsqueeze(0)
        self.__setitem__(([], 0), ((self.sos, torch.tensor(0.0, device=self.device)),
                                   None,
                                   [],
                                   ([], 0),
                                   {}))
        # if self.fwd_model is None:
        #     self.fwd_model = torch.jit.trace(self.fwd_model_ns, torch.tensor([self.model.sos, 40, 77, self.model.eos], dtype=torch.int64))
        '''{(ngram, timestamp): (("history", full score), ???, posterior, (previous_ngram, prev_time), word_dict] }'''

    def get_value_by_locality(self, timed_cache, key):
        cached_keys = np.fromiter(timed_cache.keys(), dtype=int)
        distance = np.abs(cached_keys - key)
        min_idx = np.argmin(distance)
        if distance[min_idx] <= self.cache_locality:
            return timed_cache[cached_keys[min_idx]]
        else:
            raise KeyError

    def __getitem__(self, k):
        ngram, timestamp = k
        try:
            ngram_dict = super(Cache, self).__getitem__(tuple(ngram))
        except Exception as e:
            print("Search k: {} | Keys in cache: {} ".format(k, super(Cache, self).keys()))
            raise e
        return self.get_value_by_locality(ngram_dict, timestamp)

    def __setitem__(self, k, v):
        ngram, timestamp = k
        try:
            ngram_dict = super(Cache, self).__getitem__(tuple(ngram))
            return ngram_dict.__setitem__(timestamp, v)
        except KeyError:
            return super(Cache, self).__setitem__(
                tuple(ngram), {timestamp: v})

    def __delitem__(self, k):
        ngram, timestamp = k
        return super(Cache, self).__delitem__(tuple(ngram))

    def get(self, k, default=None):
        try:
            return self.__getitem__(k)
        except KeyError:
            return default

    def setdefault(self, k, default=None):
        try:
            return self.__getitem__(k)
        except KeyError:
            return self.__setitem__(k, default)

    def pop(self, k, v=object()):
        ngram, timestamp = k
        if v is object():
            return super(Cache, self).pop(tuple(ngram))
        return super(Cache, self).pop(tuple(ngram), v)

    def update(self, mapping=(), **kwargs):
        super(Cache, self).update(self._process_args(mapping, **kwargs))

    def __contains__(self, k):
        ngram, timestamp = k
        contain_ngram = super(Cache, self).__contains__(tuple(ngram))
        if not contain_ngram:
            return False
        try:
            _ = self.__getitem__(k)
            return True
        except KeyError:
            return False

    def copy(self):
        return type(self)(self, self.model, self.dict)

    @classmethod
    def fromkeys(cls, keys, v=None):
        return super(Cache, cls).fromkeys((k for k in keys), v)

    def __repr__(self):
        return '{0}({1})'.format(
            type(self).__name__, super(Cache, self).__repr__())

    def get_state(self, k):
        return deepcopy(self.__getitem__(k)[0][0]), self.__getitem__(k)[0][1].clone()

    def get_pred(self, k, word):
        hist, hist_score = self.get_state(k)
        if word == lattice.SOS:
            score = self.zero
        else:
            if word in self.get_word_dict(k):
                return self.get_word_dict(k)[word]
            if word == lattice.EOS:
                y = torch.cat([hist, self.eos])
            else:
                mapped_word = self.acronyms.get(word, word)
                hyp = self.sp.encode_as_pieces(' ' + mapped_word)
                hyp = torch.tensor([sym2idx(self.dict, char) for char in hyp], dtype=torch.int64, device=self.device)
                y = torch.cat([hist, hyp])
            score = self.fwd_model(y)
        self.get_word_dict(k)[word] = score - hist_score
        return score - hist_score

    def fwd_model(self, y):
        with record_function('unsqueeze'):
            y_in = y[:-1].unsqueeze(0)
        with record_function('mask'):
            y_mask = target_mask(y_in, self.model.ignore_id)
        # if self.store_grad_cpu:
        #     with torch.autograd.graph.save_on_cpu():
        #         pred = self.model.decoder(y_in, y_mask, self.h, None)[0]
        #         pred = torch.nn.functional.log_softmax(pred[0], dim=1)
        #         score = torch.sum(
        #                 pred[torch.arange(pred.size(0)), y[1:]])
        # else:
        with record_function('decode'):
            pred = self.model.decoder(y_in, y_mask, self.h, None)[0]
        with record_function('logsoftmax'):
            pred = torch.nn.functional.log_softmax(pred[0], dim=1)
        with record_function('sum'):
            score = torch.sum(
                pred[torch.arange(pred.size(0)), y[1:]])
        return score
    
    def renew(self, prev_ngram, new_ngram, post):
        word = new_ngram[0][-1]
        if word in [lattice.OOV, lattice.UNK]:
            # skip oov and unk
            value = self.__getitem__(prev_ngram)
        else:
            hist, hist_score = self.get_state(prev_ngram)
            if word == lattice.SOS:
                score = self.zero
                state = (self.sos, score)
            else:
                if word == lattice.EOS:
                    y = torch.cat([hist, self.eos])
                else:
                    mapped_word = self.acronyms.get(word, word)
                    hyp = self.sp.encode_as_pieces(' ' + mapped_word)
                    hyp = torch.tensor([sym2idx(self.dict, char) for char in hyp], dtype=torch.int64, device=self.device)
                    y = torch.cat([hist, hyp])
                with record_function('forward_model'):
                    score = self.fwd_model(y)
                state = (y, score)

            self.get_word_dict(prev_ngram)[word] = score - hist_score
            value = (state, None, post, prev_ngram, {})
        self.__setitem__(new_ngram, value)

    def get_post(self, k):
        return self.__getitem__(k)[2]

    def get_prev_ngram(self, k):
        return self.__getitem__(k)[3]

    def get_word_dict(self, k):
        # this is only for ISCA subword rescoring
        return self.__getitem__(k)[4]

    def get_timestamp(self, k):
        ngram, timestamp = k
        ngram_dict = super(Cache, self).__getitem__(tuple(ngram))
        cached_keys = list(ngram_dict.keys())
        distance = [abs(timestamp - i) for i in cached_keys]
        min_idx = np.argmin(distance)
        return cached_keys[min_idx]

# ([], 0): (('', torch.tensor(0.0)), None, [], ([], 0), {})
'''{(ngram, timestamp): [("history", full score), ???, posterior, (previous_ngram, time?), word_dict] }'''

def backprop(loss, word_count, optim, keep=False):
    loss = loss/word_count # avg loss
    loss.backward(retain_graph=keep)
    optim.step()
    optim.zero_grad(set_to_none=True)
    # return loss.item()  # mean loss

# @profile_every(100)
def isca_rescore(in_lat, feat, model, char_dict, ngram, sp, cache,
                 replace=False, cache_locality=9, acronyms={}, device='cpu', optim=None, max_arcs=500):
    """Lattice rescoring with LAS model with on-the-fly lattice expansion
    using n-gram based history clustering.
    Optionally, run forward-backward before calling this function,
    so the cache can be updated based on lattice node posterior.

    :param lat: Word lattice object.
    :type lat: lattice.Lattice
    :param feat: Acoustic feature for the utterance.
    :type feat: np.ndarray
    :param model: ESPnet RNN-based LAS model.
    :type model: torch.nn.Module
    :param char_dict: Mapping from character to index.
    :type char_dict: dict
    :param ngram: Number of n-gram for history clustering.
    :type ngram: int
    :param sp: Sentencepiece model.
    :type sp: sentencepiece model object
    :param model_type: Type of the end-to-end model, one of ['las', 'tfm'].
    :type model_type: str
    :param replace: Replace existing scores if True, otherwise append.
    :type replace: bool
    :param cache_locality: Only use cache if around given number of frames.
    :type cache_locality: int
    :param acronyms: Mapping to match vocabulary.
    :type acronyms: dict
    :return: An expanded lattice with ISCA score on each arcs.
    :rtype: lattice.Lattice
    """
    # setup ngram cache
    lat = deepcopy(in_lat)
    store_grad_cpu = lat.num_arcs() > max_arcs
    # initialise expanded node & outbound arc list
    for node in lat.nodes.values():
        node.expnodes = []
        node.exparcs = []
    lat.start[1].expnodes.append(lat.start[1].subnode())

    tot_loss = torch.tensor(0.0)
    loss = torch.tensor(0.0, device=device)
    word_count = 0
    # lattice traversal
    
    with record_function("lattice_traversal"):
        for n_i in lat.nodes.values():
            for n_j in n_i.expnodes:
                for a_k_idx in n_i.exits:
                    # find the destination node n_k of arc a_k
                    n_k = lat.get_arc_dest_node(a_k_idx)
                    # find the LM state phi(h_{n_0}^{n_j}) of expanded node n_j
                    phi_nj = n_j.lmstate
                    # find a new LM state phi(h_{n_0}^{n_k}) for node n_k
                    phi_nk = ((phi_nj[0] + [n_k.sym])[-ngram:], n_k.entry)
                    # check if the destination node needs to be expanded
                    try:
                        idx = [i.lmstate[0] for i in n_k.expnodes].index(
                               phi_nk[0])
                        n_l = n_k.expnodes[idx]
                    except ValueError:
                        # create a new node for expansion
                        n_l = n_k.subnode()
                        n_l.lmstate = deepcopy(phi_nk)
                        n_k.expnodes.append(n_l)
                    new_arc = lat.arcs[a_k_idx].subarc(n_j, n_l)

                    # update cache except for the final node
                    # if n_k.sym != lattice.EOS:
                    phi_nk_post = (cache.get_post(phi_nj) + [lat.arcs[a_k_idx].post])[-ngram:]
                    # compute LM probability P(n_k|phi(h_{n_0}^{n_j}))
                    if phi_nk not in cache:
                        # create new entry in cache for unseen ngram
                        with record_function("renew_new_ngram"):
                            cache.renew(phi_nj, phi_nk, phi_nk_post)
                    else:
                        timestamp_condition = (
                            abs(cache.get_timestamp(phi_nk) - n_k.entry)
                            <= cache.cache_locality)
                        if (cache.get_prev_ngram(phi_nk)[0] == phi_nj
                            and timestamp_condition):
                            # if the previous ngram phi_nj is the same
                            # then cache hit, do not forward again
                            # note that same ngram can have different posterior
                            # because of different timestamps
                            pass
                        else:
                            posterior_condition = (
                                sum(phi_nk_post) > sum(cache.get_post(phi_nk)))
                            # if the previous ngram phi_nj is different
                            if posterior_condition or not timestamp_condition:
                                # renew the cache with higher posterior
                                with record_function("renew_seen_ngram"):
                                    cache.renew(phi_nj, phi_nk, phi_nk_post)


                    if n_k.sym in [lattice.OOV, lattice.UNK]:
                        word_count += 1
                        # loss_vec.append(0.0)
                    else:
                        with record_function("get_pred"):
                            loss = loss - cache.get_pred(phi_nj, n_k.sym)
                            word_count += 1
                    new_arc_idx = lat.add_arc(new_arc)
                    n_j.exits.append(new_arc)
                    n_l.entries.append(new_arc)
                    if word_count >= max_arcs:
                        with record_function("backprop_max"):
                            backprop(loss, word_count, optim, keep=True)
    with record_function("backprop"):
        backprop(loss, word_count, optim)
    return tot_loss
