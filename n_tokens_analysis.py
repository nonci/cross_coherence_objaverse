# PRECOMP.
from objaverse_utils import get_sampler, load_texts
from collections import OrderedDict
import pickle
FOLDER = './cross_coherence_objaverse/data/'

DATA_DIR = '/media/data2/dmercanti/datasets'
#PCs_PATH = f'{DATA_DIR}/objaverse_660K/8192_npy/'

txts = load_texts(f'{DATA_DIR}/objaverse_660K/Cap3D_automated_Objaverse_highquality.csv')
#print(len(txts))  #549922

from transformers import T5Tokenizer
import torch 

lang_model_name = 't5-11b'
t5_tokenizer = T5Tokenizer.from_pretrained(
	lang_model_name,
	model_max_length=12000,
	cache_dir='/media/data2/dmercanti/.cache/huggingface/hub/',
	legacy=False
) #

n_tk = []
max_chars =0
for _, t in txts.items():
	#t = t.lower()
	input_text = t5_tokenizer(t, return_tensors="pt")
	n_tk.append(input_text['input_ids'].shape[1])
	#if len(t)>max_chars:
	#	max_chars = len(t)

print(max(n_tk))
with open(f"{FOLDER}objaverse_n_tokens_data", 'wb') as f:
    pickle.dump(n_tk, f)

#print((torch.linalg.norm(input_text['input_ids']-input_text2['input_ids'])))

'''
# ANALYSIS:
FOLDER = './cross_coherence_objaverse/data/'

import pickle
import matplotlib.pyplot as plt
import torch 

with open(f"{FOLDER}objaverse_n_tokens_data", 'rb') as f:
    data = pickle.load(f)

plt.hist(data, 30, range=(min(data).item(), max(data).item()))
plt.grid()
plt.vlines(19, 0, 80000, colors='red')
plt.savefig(f'{FOLDER}dist.png')

print(max(data))
print(torch.quantile(data, 0.9999))  # tensor(51., dtype=torch.float64)
'''