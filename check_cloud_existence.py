from objaverse_utils import get_sampler, load_texts #, get_text_embeddings
from tqdm import tqdm
import json

ID_TEXT_FILE = '/media/data2/dmercanti/datasets/objaverse_660K/Cap3D_automated_Objaverse_highquality.csv'  # 549,922 sAMPLES
PCs_PATH = f'/media/data2/dmercanti/datasets/objaverse_660K/8192_npy/'
B_SIZE = 128

id_to_texts = load_texts(ID_TEXT_FILE)
dataset = get_sampler(id_to_texts, B_SIZE, device='cpu', tensors_path=PCs_PATH,\
            check_file=True, deterministic=True, drop_last=False, dry_run=True)


d = []
nb=0
print(len(id_to_texts))
for _, _, _, disc in tqdm(dataset, total=549922//B_SIZE):
    d += disc
    nb+=1


with open('/home/dmercanti/dev/cross_coherence_objaverse/data/not_found', 'w') as fp:
    json.dump([_.split('/')[-1] for _ in d], fp)