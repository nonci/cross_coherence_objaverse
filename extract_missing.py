'''  To extract from zipped file ZF only the clouds whose id is specified in NEEDED.  '''

import zipfile
import json
import os

BASE = '/media/data2/dmercanti/datasets/objaverse_660K/val-test/'
#ZF = f'{BASE}compressed_pcs_pt_04.zip'
# from: https://huggingface.co/datasets/tiange/Cap3D/tree/main/PointCloud_pt_zips
#NEEDED = '/media/data2/dmercanti/datasets/objaverse_660K/val-test/val_object_ids_3000.txt'
# from: https://huggingface.co/datasets/RunsenXu/PointLLM/tree/main
NEEDED = '/home/dmercanti/dev/cross_coherence_objaverse/not_found'
TO = f'{BASE}from_cap3D/'

with open(NEEDED, 'r') as nf:
    needed = [_[:_.index('_')] for _ in json.load(nf)] #nf.readlines()]

for n in range(5):
	fname = f'compressed_pcs_pt_0{n}.zip'
	if not os.path.exists(f'{BASE}{fname}'):
		os.system('cd ' + BASE + '; wget ' + ("-c " if os.path.exists(f'{BASE}{fname}' + "?download=true") else "")+r'https://huggingface.co/datasets/tiange/Cap3D/resolve/main/PointCloud_pt_zips/'+fname+r'\?download=true')
		os.system('cd ' + BASE + '; mv ' + fname + r'\?download=true ' + fname)
	else:
		print('Using existing file', n)
	with zipfile.ZipFile(f'{BASE}{fname}', 'r') as zf:
		l = [_[_.index('/')+1:_.index('.')] for _ in zf.namelist()]
		# 'Cap3D_pcs_pt/157800b903a647308a25627f134132b6.pt'
		p = [_ for _ in l if _ in needed]
		for id_ in p:
			zf.extract(f'Cap3D_pcs_pt/{id_}.pt', path=TO)
		print(len(p), f'files extracted from file: {n}-th')
	os.system("cd " + BASE + "; rm " + fname )  # + r'\?download=true'
 
 # 56064 files extracted from file: 4-th
 # 1 files extracted from file: 2-th
 
 # for f in $(ls 00*); do mv $f "${f#00}"; done