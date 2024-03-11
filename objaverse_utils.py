import torch, os, sys, csv, random
from collections import OrderedDict, defaultdict
import numpy as np

DEFAULT_DEV = 'cpu'

'''
class sample:
	def __init__(self, id_, text, text_embed=None, cloud=None, is_cloud_encoded=True) -> None:
		self.id = id_
		self.txt = text
		self.txt_embed = text_embed
		self.cloud = cloud
		self.is_cloud_encoded = is_cloud_encoded

def load_texts(where):
	# Loads texts and saves them into "id_to_texts".
	id_to_texts = OrderedDict()
	with open(where, "r") as f:
		reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_MINIMAL, strict=True, )
		for row in reader:
			id_to_texts[row[0]] = row[1]
	return id_to_texts
'''

def is_gpu_avail(dev):
	a,b = torch.cuda.mem_get_info(device=dev)
	return ((b-a)/2**30)<1.  # less than 1GiB reserved

def avail_gpus(as_int=True):
	devs = [(i if as_int else torch.cuda.device(i)) for i in range(torch.cuda.device_count())]
	return [d for d in devs if is_gpu_avail(d)]

def pc_norm_colors(pc):
	raise NotImplementedError


def pc_norm_batched(pc):
	"""
	Similar to pointllm.data.pc_norm, but acts with torch tensors, batched.
	pc: BxNxC, return BxNxC
	"""
	# for debugging: pc_norm(pc[0].cpu().numpy()) (m is a single value etc.)
	xyz = pc[:, :, :3]
	other_feature = pc[:, :, 3:]
	centroid = torch.mean(xyz, axis=1, keepdim=True)
	xyz = xyz - centroid
	m = torch.max(torch.sqrt(torch.sum(xyz ** 2, axis=1)), dim=1).values.view((pc.shape[0],1,1))
	xyz = xyz / m
	return torch.concat((xyz, other_feature), axis=2)


def txt_embedding(text, tokenizer, model, device=DEFAULT_DEV, max_len=75, force_lower=False):  #truncate_to=1024
	'''Given a model and a tokenizer, computes the embedding of a text.'''
	if force_lower: text = text.lower()
	input_text = tokenizer(text, return_tensors="pt", padding='max_length', max_length=max_len, truncation=True).to(device)
	#input_text = tokenizer(text, return_tensors="pt").to(device)
	out = model(input_ids=input_text.input_ids, attention_mask=input_text.attention_mask, ) #output_hidden_states=True
	last_hidden_state = out.last_hidden_state.detach()  # torch.Size([1, <seq length>, 1024])
	text_embed = last_hidden_state.squeeze(0)
	#if text_embed.shape[1] > truncate_to:
	#	truncated = True
	#	text_embed = text_embed[:, :truncate_to]       # truncate to max_length => [max_length, 1024]
	assert text_embed.shape == (max_len, 1024)
	return text_embed


def get_sampler(dataset: OrderedDict, locator, n: int, drop_last=False, \
	device=DEFAULT_DEV, fallback_locator=None, deterministic=True, dry_run=False):
	''' Return iterator that gives a batch at each iteration.

	Each batch is a tuple:
		ids: list,
		clouds: torch.tensor n x 8192 x 6,
		texts: list
		(discarded_ids): list  (only if check_file)
	Param.s:
		dataset: dict of ids and texts,
		n: batch size
		locator: fun. taking id (str) and returning a path (str)
		fallback_locator: idem, but is used when the former fails; None and 'ignore_missing' are also allowed, where the 1st will raise an error if the file doesn't exist.
  	'''
	if type(dataset) is not OrderedDict:
		print('WARNING: passed dataset to get_sampler of type not OrderedDict; order may be wrong!')
	clouds = []
	texts = []
	ids = []
 
	if not deterministic:
		dataset = shuffle_dict(dataset)
	
	i=0
	for id_, txt in dataset.items():
		fpath = locator(id_)
		if not os.path.exists(fpath):
			if fallback_locator=='ignore_missing':
				continue
			elif fallback_locator:
				fpath = fallback_locator(id_)
		if not dry_run:
			clouds.append(torch.from_numpy(np.load(fpath)) if fpath.endswith('.npy') else torch.load(fpath))
			# clouds[-1]: [8192,6] vs. [6,16384]
			if clouds[-1].shape == torch.Size([6,16384]):
				clouds[-1] = clouds[-1].view((6,8192,2))[:,:,0].view(6, 8192).T # keep half of the points
				#clouds[-1] = pc_norm_batched(clouds[-1].view((1, *clouds[-1].shape))).view((8192, 6))
			texts.append(dataset[id_])
			ids.append(id_)
		else: assert os.path.exists(fpath)
		i+=1
		if not i%n:
			if dry_run:
				yield ([], [], [])
			else:
				yield (ids, torch.stack(clouds, axis=0).to(device), texts, )
				clouds, texts, ids = [], [], []
   
	if clouds and not drop_last:
		yield (ids, torch.stack(clouds, axis=0).to(device), texts,)


def get_sampler_hard_ds(ids_and_texts: dict, clouds_folder, txt_emb_folder, device=DEFAULT_DEV):
    ''' ids_and_texts has keys 0,1,... and values: dicts with keys: 'txt', 'id', 'knn-txt', 'knn-id'
    This sampler immplements the protocol used with HST in Amaduzzi et al., 2023 '''
    for i in range(len(ids_and_texts)):
        d = ids_and_texts[i]
        gt_txt, id_, knn_txt, knn_id = d['txt'], d['id'], d['knn-txt'], d['knn-id']
        gt = random.randint(0,1)
        
        txts = [knn_txt, gt_txt] if gt else [gt_txt, knn_txt]
        ids  = [knn_id, id_] if gt else [id_, knn_id]
        clouds = torch.stack([
            torch.tensor(np.load(f'{os.path.join(clouds_folder, id_)}_8192.npy')).to(device),
            torch.tensor(np.load(f'{os.path.join(clouds_folder, knn_id)}_8192.npy')).to(device)])
        txt_es = torch.stack(get_text_embeddings(ids, txt_emb_folder, device, pad_to=None))
        
        yield (txts, ids, clouds[[gt, 1-gt],:,:], txt_es, gt)
        #torch.stack(get_text_embeddings(orig_ids, TXT_EMB_PATH, DEV_TEST, pad_to=None))
    
    

def get_single_cloud_safe(id_, tensors_path: str, device=DEFAULT_DEV):
	''' Loads a pre-computed shape embedding from disk, returning None if the file doesn't exist. '''
	fname = f'{os.path.join(tensors_path, id_)}_8192.npy'
	return torch.tensor(np.load(fname)).to(device) if os.path.exists(fname) else None

def get_single_cloud(id_, tensors_path: str, device=DEFAULT_DEV):
	''' Loads a pre-computed shape embedding from disk. '''
	return torch.tensor(np.load(f'{os.path.join(tensors_path, id_)}_8192.npy')).to(device)


def load_pt_encoder(config_file, weights_file, pointllm_dir, device=DEFAULT_DEV, use_max_pool=None):
	''' Loads pointBert pre-trained point encoder (in eval mode). '''
	
	from collections import OrderedDict
	sys.path.append(pointllm_dir)
	from pointllm.model import PointTransformer
	from pointllm.utils import cfg_from_yaml_file
	#from pointllm.data import pc_norm
	del sys.path[-1]
 
	point_bert_config = cfg_from_yaml_file(config_file)
	point_bert_config.model.point_dims = 6  # Assuming use_colors is True
	if use_max_pool is None:
		use_max_pool = getattr(point_bert_config.model, "use_max_pool", False) # * default is false
	
	point_encoder = PointTransformer(point_bert_config.model, use_max_pool=use_max_pool).to(device)
	point_encoder.eval()
	ckpt = torch.load(weights_file, map_location=device)

	state_dict = OrderedDict()
	for k, v in ckpt.items():
		if k.startswith('module.point_encoder.'):
			state_dict[k.replace('module.point_encoder.', '')] = v
	point_encoder.load_state_dict(state_dict)

	#for p in point_encoder.parameters():
	#	p.requires_grad = False

	return point_bert_config, point_encoder


def load_t5_text_encoder(device=DEFAULT_DEV):
	''' Loads t5 text encoder; see:
		https://huggingface.co/t5-11b
		https://huggingface.co/docs/transformers/tokenizer_summary '''

	from transformers import T5EncoderModel, T5Tokenizer
	lang_model_name = 't5-11b'
	t5_tokenizer = T5Tokenizer.from_pretrained(
		lang_model_name,
		model_max_length=512,
	 	cache_dir='/media/data2/dmercanti/.cache/huggingface/hub/',
		legacy=False
	) # 
	t5 = T5EncoderModel.from_pretrained(lang_model_name, cache_dir='/media/data2/dmercanti/.cache/huggingface/hub/') #'/home/dmercanti/.cache/huggingface/hub/'
	t5 = t5.to(device)
	return t5_tokenizer, t5


def load_texts(where):
	''' Loads texts and saves them into a "id_to_texts" dict. '''
	id_to_texts = OrderedDict()
	with open(where, "r") as f:
		reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_MINIMAL, strict=True, )
		for row in reader:
			id_to_texts[row[0]] = row[1]
	return id_to_texts


def shuffle_dict(D):
	keys = list(D.keys())
	random.shuffle(keys)
	d = OrderedDict([(k, D[k]) for k in keys])
	return d

def deterministic_shuffle_dict(D, seed=0):
	state = random.getstate()
	random.seed(seed)
	shuffled = shuffle_dict(D)
	random.setstate(state)
	return shuffled


def split(dataset, *ratios , shuffle=False, check_partition=False):
	''' dataset: indexable seq. or dict to be split 
	ratios: in [0., 1.]; have to sum up to 1 if check_partition is True.
	shuffle can be False or a function that returns the shuffled dataset; for lists use:
	shuffle = lambda _: random.shuffle(_) or _; 
	for dictionaries use shuffle_dict. '''
	from itertools import accumulate, groupby
	if shuffle:
		dataset = shuffle(dataset)
	if check_partition:
		assert (abs(sum(ratios)-1.)<1.e-10)
	tot = len(dataset)
	indexes = list(accumulate([0, *[round(r*tot) for r in ratios]]))
	if check_partition and indexes[-1]!=len(dataset):
		print('Split result is approximated!')
	indexes[-1] = len(dataset)
	if type(dataset) in (dict, OrderedDict):
		k_list_sp = split(list(dataset.keys()), *ratios, shuffle=False)  # lambda _:random.shuffle(_) or _
		return [ OrderedDict([(k, dataset[k]) for k in v]) for v in k_list_sp ]
	else:
		return [dataset[indexes[i]:indexes[i+1]] for i in range(len(indexes)-1)]


def get_text_embeddings(ids, tensors_path: str, device=DEFAULT_DEV, pad_to=None):
	''' Loads pre-computed text embeddings from disk.
	ids is a list of str or a single str. '''
	if type(ids) is not list:
		ids = [ids]
	text_embeds = [torch.load(f'{os.path.join(tensors_path, id_)}.pt').to(device) for id_ in ids]
	if pad_to:
		raise NotImplementedError  # Incorrect to pad with zeros here!
		'''
		for i, text_embed in enumerate(text_embeds):
			shp0 = text_embed.shape[0]
			if shp0 < pad_to:
				# add zeros at the end of text embed to reach pad_to     
				pad = torch.zeros(pad_to - shp0, text_embed.shape[1])
				pad = pad.to(device)
				text_embeds[i] = torch.cat((text_embed, pad), dim=0)
			else:
				# truncate
				text_embeds[i] = text_embed[:pad_to, :]
    	'''	   
	return text_embeds

#def render_cloud(c):
#	pcd_data = o3d.data.PLYPointCloud()
#	pcd = o3d.io.read_point_cloud(pcd_data.path)
#	o3d.visualization.draw_geometries([pcd])


class Stats_handler:
    def __init__(self, optimizer=None) -> None:
        self.epoch_loss = 0.
        self.epoch_corrects = defaultdict(lambda:0.)
        self.o = optimizer
    
    def update_stats(self, logits, col_labels, row_labels, epoch_loss_val):
        preds_ok_0 = torch.max(logits, dim=0)[1] == col_labels  # is each SHAPE associated to the right text?
        preds_ok_1 = torch.max(logits, dim=1)[1] == row_labels # is each TEXT associated to the right shape?
        preds_ok = torch.logical_and(preds_ok_0, preds_ok_1).count_nonzero()
        self.epoch_loss += epoch_loss_val
        self.epoch_corrects['tot'] += torch.sum(preds_ok)
        self.epoch_corrects['shp'] += torch.sum(preds_ok_0)  # CORRECTED
        self.epoch_corrects['txt'] += torch.sum(preds_ok_1)  # CORRECTED

    def get_and_reset_stats(self, n_examples):
        lr = self.o.param_groups[0]['lr'] if self.o else None
        epoch_loss = self.epoch_loss / n_examples
        epoch_acc = self.epoch_corrects['tot'].double() / n_examples
        print('rel.acc: ', epoch_acc, '- abs.acc.:', epoch_acc*n_examples)
        epoch_txt_acc = self.epoch_corrects['txt'].double() / n_examples
        epoch_shp_acc = self.epoch_corrects['shp'].double() / n_examples
        self.__init__(self.o)
        return lr, epoch_loss, epoch_acc, epoch_txt_acc, epoch_shp_acc
    
def remove_clouds_in(od, from_):
    import json
    with open(from_, 'rb') as f:
        not_found = json.load(f)
    for nff in not_found:
        del od[nff.split('_')[0]]
    return od