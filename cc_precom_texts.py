''' This script pre-computes text embeddings for Cap3D descriptions found in ID_TEXT_FILE.
	Embeddings are not truncated and are saved in EMB_DIR.
'''

base_dir = '/home/dmercanti/dev'
import torch, sys, os, tqdm
sys.path.append(f'{base_dir}/shapeglot_text2shape/')
from objaverse_utils import txt_embedding, load_texts, load_t5_text_encoder, is_gpu_avail, avail_gpus


#ID_TEXT_FILE = '/media/data2/dmercanti/datasets/objaverse_660K/Cap3D_automated_Objaverse.csv'  # 661577 samples
ID_TEXT_FILE = '/media/data2/dmercanti/datasets/objaverse_660K/Cap3D_automated_Objaverse_highquality.csv'  # 549922 samples
EMB_DIR = '/media/data4/dmercanti/cap3D_txt_embeddings/'
DEV = avail_gpus()[0]
B_SIZE = 4


def main():
	id_to_texts = {}
	device = torch.device(DEV if torch.cuda.is_available() else "cpu")
	print('Using device:', device)
	assert is_gpu_avail(device)
	
	print(f'Loading texts from {ID_TEXT_FILE}')
	id_to_texts = load_texts(ID_TEXT_FILE)

	print(end='Loading t5 text encoder...', flush=True)
	tokenizer, txt_encoder = load_t5_text_encoder(device=device)
	print('ok')
 
	for id_, text in tqdm.tqdm(id_to_texts.items()):
		txt_emb = txt_embedding(text, tokenizer, txt_encoder, device=device,\
      		max_len=75) # 75 is quantile 1.; 51 is quantile 0.9999
		torch.save(txt_emb, os.path.join(EMB_DIR, f'{id_}.pt'))
  		#print(txt_emb.shape)


if __name__=='__main__':
    main()
    
    
'''
text_embed = torch.load(text_embed_path, map_location=device)
if text_embed.shape[0] > max_length:
	count_trunc += 1
	text_embed = text_embed[:max_length]       # truncate to max_length => [max_length, 1024]
'''
