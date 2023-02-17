import k2
import openfst
from utils import kaldiLatticeIterator
from lattice import Lattice

# it = kaldiLatticeIterator('./examples', suffix_in='.lat')
# examples = ["AMI_EN2001a_H00_MEE068_0000557_0000594", "AMI_EN2001a_H00_MEE068_0001109_0001553",
#       "AMI_EN2001a_H00_MEE068_0001895_0001926", "AMI_EN2001a_H00_MEE068_0002141_0002577",
#       "AMI_EN2001a_H00_MEE068_0002809_0003300", "AMI_EN2001a_H00_MEE068_0003300_0003961",
#       "AMI_EN2001a_H00_MEE068_0003961_0004004", "AMI_EN2001a_H00_MEE068_0004224_0004746",
#       "AMI_EN2001a_H00_MEE068_0005664_0005689", "AMI_EN2001a_H00_MEE068_0008002_0008038"]
# for ex in examples:
#     klat = Lattice(it.lattice_dict[ex], file_type='kaldi')
#     hlat = Lattice('./examples/{}.lat.gz'.format(ex))
#     kark = set([str(x) for x in klat.arcs])
#     hark = set([str(x) for x in hlat.arcs])
#     if hark == kark:
#         print(ex)
#     else:
#         d1 = kark.difference(hark)
#         d2 = hark.difference(kark)
#         print(ex + ' wrong')


with open('examples/1.lat.sym') as f:
    ln = f.readlines()

lat = {}
tmp_lat = []
uttid_found = False
for line in ln:
    line = line.strip()
    if not uttid_found:
        uttid = line
        uttid_found = True
    else:
        if len(line) == 0:
            lat[uttid] = tmp_lat
            tmp_lat = ["0 1 -2 1 _"]
            uttid_found = False
        elif len(line.split(' ')) == 1:
            tmp_lat.append(line)
        else:
            tmp_lat
            ls = line.split(' ')
            weights = ls[-1].split(',')
            w = float(weights[0]) + float(weights[1])
            t = len(weights[2].split('_'))
            tmp_lat.append("{} {} {} {} {}".format(int(ls[0]) + 1, int(ls[1]) + 1, t, w))

fsa = k2.Fsa.from_openfst('\n'.join(lat["AMI_EN2001a_H00_MEE068_0003961_0004004"]), aux_label_names=['frames'])
noeps = k2.remove_epsilon(fsa)
noeps.frames = noeps.frames.sum()
print(fsa.to_str())
print(noeps.to_str())