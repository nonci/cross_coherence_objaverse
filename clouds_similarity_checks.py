'''
Data preparation, see:
	https://github.com/OpenRobotLab/PointLLM/tree/a546f4250436f4c8241da7ade24c5f42ad5d5111?tab=readme-ov-file#data-preparation
Use:
	https://github.com/OpenRobotLab/PointLLM/blob/master/pointllm/model/pointllm.py
Weights, see:
	https://github.com/LilRedWu/PointBert/blob/main/README.md

Note that easydict is required (from utils.py in poinllm).
PointLLM:
  ... point clouds to have 8192 points, as our model is trained on such point clouds
  The point encoder outputs m = 513 point features, each with c = 384 dimensions.
'''

import torch, sys, os, stat, random #, json
from torch.nn.functional import cosine_similarity
import numpy as np

# For reproducibility:
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
np.random.seed(0)
# rst = torch.random.get_rng_state()
random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

base_dir = '/home/dmercanti/dev'

sys.path.append(f'{base_dir}/shapeglot_text2shape/')
from custom_utils import visualize_pointcloud, plot_cloud_p3d

from objaverse_utils import pc_norm_batched, txt_embedding, load_texts, \
	get_sampler, load_pt_encoder, load_t5_text_encoder, sample, get_single_cloud_safe


ID_TEXT_FILE = '/media/data2/dmercanti/datasets/objaverse_660K/Cap3D_automated_Objaverse.csv'  # 661577 samples
#PBERT_CONFIG_FILE = f'{base_dir}/PointLLM/pointllm/model/pointbert/PointTransformer_base_8192point.yaml'  # default
PBERT_CONFIG_FILE = f'{base_dir}/PointLLM/checkpoint_point_encoder/PointTransformer_tiny_8192point.yaml'  # default
PBERT_WW = f'{base_dir}/PointLLM/checkpoint_point_encoder/pointBert_ww_Tiny/checkpoint_7573.pt'
#PBERT_WW = f'{base_dir}/PointLLM/checkpoint_point_encoder/Point-BERT.pth' #point_bert_v1.1.pt
PCs_PATH = '/media/data2/dmercanti/datasets/objaverse_660K/8192_npy/'
USE_STORED_TEXTS = True
DEV = "cpu" #"cuda:0"
B_SIZE = 4 # if txt.enc not loaded on GPU, use 32
MPOOL = True  # Tested with similarity_check only! 


def main():
	id_to_texts = {}
	#id_to_enc_shape = {}
	device = torch.device(DEV if torch.cuda.is_available() else "cpu")
	print('Using device:', device)
	
	print(f'Loading texts from {ID_TEXT_FILE}')
	id_to_texts = load_texts(ID_TEXT_FILE)
		
	print(f"Loading PointBERT; config from {PBERT_CONFIG_FILE}.")
	_, point_encoder = load_pt_encoder(
    	PBERT_CONFIG_FILE,
    	PBERT_WW,
     	f'{base_dir}/PointLLM/',
      	device=device,
       	use_max_pool=MPOOL  # use True to interpret the result as global without further proc.
    )

	# if not USE_STORED_TEXTS:
	#	print(end='Loading t5 text encoder...', flush=True)
	#	tokenizer, txt_encoder = load_t5_text_encoder(device=device)
	#	print('ok')
	# txt_emb = txt_embedding(texts[i], tokenizer, txt_encoder, device='cpu').to(device)
	# dataset[ids[i]] = sample(ids[i], texts[i], txt_emb, encoded_cloud[i])

	sampler = get_sampler(id_to_texts, B_SIZE, device=device, tensors_path=PCs_PATH)
	
 
	if sys.argv[1]=='store_enc_clouds':
		''' For the entire ds, around 4 TB is required! '''
		stop_at = int(sys.argv[2])
		for b,(ids, c, _) in enumerate(sampler):
			if stop_at>0 and b*B_SIZE >= stop_at:
				break
			with torch.no_grad():
				c = pc_norm_batched(c) # Normalizes into a sphere r=1; expects bxNxC, returns the same
				encoded_cloud = point_encoder(c)
				for i in range(len(ids)):
					torch.save(encoded_cloud[i],\
         				f'/media/data2/dmercanti/tensors/objaverse_clouds/{ids[i]}_encoded.pt')
			print(end=f'\rbatch: {b}', flush=True)
   
	
	if sys.argv[1]=='most_similar':
		use_colors = True if len(sys.argv)==2 else (sys.argv[2]!='no_colors')
		print('use_colors is ', use_colors)
		while True:
			id_1 = input('id: ')
			c1 = get_single_cloud_safe(id_1, PCs_PATH, device=device)
			if c1 is None: continue
			#nc1 = pc_norm_batched(c1.view(1, *c1.shape))
			nc1=c1.view(1, *c1.shape)
			if not use_colors: nc1[:,:,3:] = 0.
			ec1 = point_encoder(nc1).view((768,))
			discarded = []
   
			max_found = {'id': [], 'val': []}
			for b, (ids, c, _, disc) in enumerate(sampler):
				for i, id_2 in enumerate(ids):
					#nc2 = pc_norm_batched(c[i].view(1, *c[i].shape))
					nc2=c[i].view(1, *c[i].shape)
					if not use_colors: nc2[:,:,3:] = 0.
					ec2 = point_encoder(nc2).view((768,))
					cos_sim = torch.nn.functional.cosine_similarity(ec1, ec2, dim=0)
					if not max_found['val'] or (cos_sim > max_found['val'][-1] and id_1!=id_2):
						max_found['val'].append(cos_sim.item())
						max_found['id'].append(id_2)
				discarded += disc
					
				print(end=f'\rCUR.MAX SIM: {max_found["id"][-1]}  (val: {max_found["val"][-1]:.3f})  -  checked ~ {b*B_SIZE/1000:.1f} K     ')
			print(max_found)
			print(discarded)
 
	if sys.argv[1]=='similarity_check':
		NORMALIZE = False # use False!
		mode = os.fstat(0).st_mode
		batch_mode =  stat.S_ISFIFO(mode) or stat.S_ISREG(mode)
		use_colors = True if len(sys.argv)==2 else (sys.argv[2]!='no_colors')
		print('use_colors is ', use_colors)
		print('max_pool is ', point_encoder.use_max_pool)
		print('normalize is ', NORMALIZE)
		
		while True:
			try:
				id1 = input('' if batch_mode else 'id 1: ')
				id2 = input('' if batch_mode else 'id 2: ')
			except (EOFError, KeyboardInterrupt):
				print()
				break
   
			c1 = get_single_cloud_safe(id1, PCs_PATH, device=device)
			c2 = get_single_cloud_safe(id2, PCs_PATH, device=device)
			if c1 is None or c2 is None:
				print('file not found', c1==-1, c2==-1)
				continue
			
			if NORMALIZE:
				nc1 = pc_norm_batched(c1.view(1, *c1.shape))
				nc2 = pc_norm_batched(c2.view(1, *c2.shape))
			else:
				nc1 = c1.view(1, *c1.shape)
				nc2 = c2.view(1, *c2.shape)
    
			if not use_colors:
				#nc1[:,:,3:] = nc1[:,:,3:]*0.5
				#nc2[:,:,3:] = nc2[:,:,3:]*0.5
				nc1[:,:,3:] = 0.
				nc2[:,:,3:] = 0.

			with torch.no_grad():
				torch.manual_seed(0)
				ec1 = point_encoder(nc1) # consistent, reproducible behav.
				torch.manual_seed(0)
				ec2 = point_encoder(nc2) # consistent, reproducible behav.

			if MPOOL:
				s3_cos = cosine_similarity(ec1.view((768,)), ec2.view((768,)), dim=0)
			else:
				s3_cos = cosine_similarity(ec1.flatten(), ec2.flatten(), dim=0)
    
			if batch_mode:
				print(f'{s3_cos.item():.3f}')
			else:
				print(f'after  encoding: {s3_cos.item():.3f} (COS.)')
			
			#print(f'{c1.shape} --norm--> {nc1.shape} --enc--> {ec1.shape}')


	if sys.argv[1]=='view_clouds':
		save = lambda c,n,t,**aa: visualize_pointcloud(
			c.cpu(),
			filename=os.path.join(base_dir, 'cross_coherence_objaverse', 'figures', n),
			marker = ('.', 1), # .,o
			title=t[:70]+'\n'+t[70:],
			verbose=True,
			**aa
		)
		stop_at = int(sys.argv[2])
		for b,(ids, c, texts) in enumerate(sampler):
			if stop_at>0 and  b*B_SIZE >= stop_at:
				break
			with torch.no_grad():
				c = pc_norm_batched(c) # Normalizes into a sphere r=1; expects bxNxC, returns the same
				encoded_cloud = point_encoder(c)
			#plot_cloud_p3d(clouds[0])
			for i in range(len(ids)):
				save(c[i], f'{b*B_SIZE+i}_xyz.png', texts[i])
				save(c[i], f'{b*B_SIZE+i}_yxz.png', texts[i], flip='yxz')
				save(c[i], f'{b*B_SIZE+i}_zxy.png', texts[i], flip='zxy')
    

if __name__ == '__main__':
	main()