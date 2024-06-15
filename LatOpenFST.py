#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
import re
import openfst_python as fst


# In[2]:


with gzip.open('tdnn_1d_sp/decode_ihm_dev_fisher_tgpr_b8.0_d75/1/AMI_ES2011a_H00_FEE041_0003714_0003915.lat.gz', 'rb') as f:
    w_lm = f.read()


# In[3]:


with gzip.open('tdnn_1d_sp/decode_ihm_dev_fisher_tgpr_b8.0_d75_nolm/1/AMI_ES2011a_H00_FEE041_0003714_0003915.lat.gz', 'rb') as f:
    no_lm = f.read()



lattice_info = {'nodes': {}, 'arcs': {}}
for line in no_lm.decode().split('\n'):
    if line:
        if line[0] == 'N':
            m = re.match('N=(\d+)\s+L=(\d+)', line)
            lattice_info['nodes_n'] = int(m.groups()[0])
            lattice_info['arcs_n'] = int(m.groups()[1])
        elif line[0] == 'I':
            m = re.match('I=(\d+)\s+t=(\d+\.\d+)\s+W=(!NULL||<.?s>|[A-Za-z]+)\s+v=(\d+)', line)
            node_i = int(m.groups()[0])
            lattice_info['nodes'][node_i] = {'t': float(m.groups()[1]), 'label': m.groups()[2],
                                             'v': int(m.groups()[3])}
        elif line[0] == 'J':
            m = re.match('J=(\d+)\s+S=(\d+)\s+E=(\d+)\s+a=(-?\d+.\d+)\s+l=(-?\d+.\d+)\s+r=(-?\d+.\d+)', line)
            arc_i = int(m.groups()[0])
            lattice_info['arcs'][arc_i] = {'start': int(m.groups()[1]), 'end': int(m.groups()[2]),
                                           'a_wt': float(m.groups()[3]), 'l_wt': float(m.groups()[4]),
                                           'r_wt': float(m.groups()[5])}


# In[6]:


syms = fst.SymbolTable()

f_htk = fst.Fst()
for i in range(lattice_info['nodes_n']):
    f_htk.add_state()
    syms.add_symbol(lattice_info['nodes'][i]['label'])
for i in range(lattice_info['arcs_n']):
    info = lattice_info['arcs'][i]
    label = syms.find(lattice_info['nodes'][info['start']]['label'])
    f_htk.add_arc(info['start'], fst.Arc(label, label, info['a_wt'], info['end']))
f_htk.set_input_symbols(syms)
f_htk.set_output_symbols(syms)
f_htk.set_start(0)
end = [x for x in lattice_info['nodes'].keys() if lattice_info['nodes'][x]['label'] == '</s>']
f_htk.set_final(end[0])


# In[7]:

# In[9]:


with open('tdnn_1d_sp/decode_ihm_dev_fisher_tgpr_b8.0_d75_nolm/kaldi/1.lat', 'rb') as f:
    full_kaldi = f.readlines()


# In[10]:


start = full_kaldi.index(b'AMI_ES2011a_H00_FEE041_0003714_0003915 \n')
end = full_kaldi[start:].index(b'\n')


# In[11]:


kaldi_no_lm = b''.join(full_kaldi[start:start+end])


# In[13]:


lat = []
for line in kaldi_no_lm.decode().split('\n'):
    line = line.strip().split(' ')
    weight = None
    if len(line) > 1:
        weights = line[-1].split(',')
        weight = float(weights[0]) + float(weights[1])
        frames = len(weights[2].split('_'))
    if weight is not None:
        lat.append([int(line[0]), int(line[1]), line[2], weight, frames*0.03])
    else:
        lat.append(line)


# In[14]:


f_k = fst.Fst()
k_syms = fst.SymbolTable()
n_states = max([x[0] for x in lat if isinstance(x[0], int)])
for i in range(n_states + 1):
    f_k.add_state()
for info in lat:
    if len(info) > 1:
        label = k_syms.add_symbol(info[2])
        f_k.add_arc(info[0], fst.Arc(label, label, -(info[3]), info[1]))
f_k.set_input_symbols(k_syms)
f_k.set_output_symbols(k_syms)
f_k.set_start(0)
f_k.set_final(n_states - 1)


# In[15]:


# In[ ]:


fkd = fst.determinize(f_k)


# In[ ]:




