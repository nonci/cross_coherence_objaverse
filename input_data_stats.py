from objaverse_utils import get_sampler, load_texts, split, shuffle_dict
from collections import OrderedDict
import re, os, json, pickle, torch, pprint


DATA_DIR = '/media/data2/dmercanti/datasets'
PCs_PATH = f'{DATA_DIR}/objaverse_660K/8192_npy/'

od = load_texts(f'{DATA_DIR}/objaverse_660K/Cap3D_automated_Objaverse_highquality.csv')
print('ORIG SIZE IS:', len(od))

# We only keep exis. ids
with open('./cross_coherence_objaverse/data/not_found', 'rb') as f:
	not_found = json.load(f)
for nff in not_found:
    #if nff in od:
	del od[nff.split('_')[0]]
print('NEW SIZE IS:', len(od))

train_txt, val_txt, test_txt = split(od, 0.9, 0.05, 0.05, shuffle=shuffle_dict)
for v in train_txt, val_txt, test_txt:
    print('split len', len(v))
    # train: 444471
    # val, test: 24693

locator = lambda id_: f'{os.path.join(PCs_PATH, id_)}_8192.npy'
#def fl(id_):
#    alt_path = '/media/data2/dmercanti/datasets/objaverse_660K/val-test/from_cap3D/Cap3D_pcs_pt/'
#    new = re.sub(r"^00(.*)", r"\1", id_)  # remove leading zeros (two of them)
#    return f'{alt_path}{new}.pt'

''' Present in PointLLM code, but not applied?
def pc_norm(pc):
    import numpy as np
    """ pc: NxC, return NxC """
    xyz = pc[:, :3]
    other_feature = pc[:, 3:]

    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    xyz = xyz / m

    pc = np.concatenate((xyz, other_feature), axis=1)
    return pc
'''

data = {
	'min_c': +9999,
	'max_c': -9999,
	'min_s': +9999,
	'max_s': -9999,
	'min_norm': +9999,
	'max_norm': -9999,
	'centroid_max_abs': torch.zeros((3,)).to('cuda:1')
}

for t in get_sampler(od, locator, 1, deterministic=False, device='cuda:1'):  #dry_run=True, fallback_locator=fl
    colors = t[1][0][:, 3:]
    shape_  = t[1][0][:, :3]
    if torch.min(colors)<data['min_c']: data['min_c']=torch.min(colors)
    if torch.min(shape_)<data['min_s']: data['min_s']=torch.min(shape_)
    if torch.max(colors)>data['max_c']: data['max_c']=torch.max(colors)
    if torch.max(shape_)>data['max_s']: data['max_s']=torch.max(shape_)
    n = torch.norm(shape_, p=2, dim=1)   # More sense with shape_
    if torch.max(n)>data['max_norm']:
        data['max_norm']=torch.max(n)
    if torch.min(n)<data['min_norm']:
        data['min_norm']=torch.min(n)
    c = torch.abs(torch.mean(shape_, axis=0))
    data['centroid_max_abs'] = torch.maximum(c, data['centroid_max_abs'])  # element-w.max.
    
pprint.pprint(data, width=50)

'''
ORIG SIZE IS: 549922
NEW SIZE IS: 493857
split len 444471
split len 24693
split len 24693
{'centroid_max_abs': tensor([0.6197, 0.7993, 1.4480]),
 'max_c': tensor(1.),
 'max_norm': tensor(2.3186),
 'max_s': tensor(1.2403),
 'min_c': tensor(0.),
 'min_norm': tensor(1.3193e-05),
 'min_s': tensor(-2.2990)
 }
 
for both shp and color => {'min_norm': tensor(1.3193e-05), 'max_norm': tensor(2.8941)}
'''


'''
with open("/home/dmercanti/dev/cross_coherence_objaverse/data/objaverse_metadata", 'rb') as f:
	md = pickle.load(f)
 
print(md['22122a47684e461099495bcc1de6d9d3'])

with open("/media/data2/dmercanti/datasets/objaverse_660K/val-test/modelnet40_test_8192pts_fps.dat","rb") as f:
    test_data = pickle.load(f)
    
print(len(test_data))


OLD GET_SAMPLER:
def get_sampler(dataset: OrderedDict, n: int, tensors_path: str, drop_last=False, device=DEFAULT_DEV, check_file=True, deterministic=True, dry_run=False):
	assert type(dataset) is OrderedDict
	clouds = []
	texts = []
	ids = []
	discarded = []
 
	if not deterministic:
		keys = list(dataset.keys())
		random.shuffle(keys)
		d = OrderedDict([(k, dataset[k]) for k in keys])
		dataset = d
	
	i=0
	for id_, txt in dataset.items():
		fpath = f'{os.path.join(tensors_path, id_)}_8192.npy'
		if check_file and not os.path.exists(fpath):
			discarded.append(fpath)
			continue
		if not dry_run:
			clouds.append(np.load(fpath))
			texts.append(dataset[id_])
			ids.append(id_)
		i+=1
		if not i%n:
			if dry_run:
				yield ([], [], [], *([discarded] if check_file else []))
			else:
				yield (ids, torch.from_numpy(np.stack(clouds, axis=0)).to(device), texts, *([discarded] if check_file else []))
				clouds, texts, ids = [], [], []
			discarded = []
   
	if clouds and not drop_last:
		yield (ids, torch.from_numpy(np.stack(clouds, axis=0)).to(device), texts, *([discarded] if check_file else []))
	elif discarded and not drop_last:
		yield ([], [], [], *([discarded] if check_file else []))

'''