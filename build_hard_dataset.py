import torch, os, time, random, sys
from tqdm import tqdm
from collections import defaultdict
from torch_cluster import knn_graph, radius_graph
from objaverse_utils import pc_norm_batched, load_texts, split, deterministic_shuffle_dict,\
    get_sampler, load_pt_encoder, load_t5_text_encoder, get_single_cloud_safe, remove_clouds_in


base_dir = '/home/dmercanti/dev'
ID_TEXT_FILE = '/media/data2/dmercanti/datasets/objaverse_660K/Cap3D_automated_Objaverse_highquality.csv'  # 661577 samples
#PBERT_CONFIG_FILE = f'{base_dir}/PointLLM/pointllm/model/pointbert/PointTransformer_base_8192point.yaml'  # default
PBERT_CONFIG_FILE = f'{base_dir}/PointLLM/checkpoint_point_encoder/PointTransformer_tiny_8192point.yaml'  # default
PBERT_WW = f'{base_dir}/PointLLM/checkpoint_point_encoder/pointBert_ww_Tiny/checkpoint_7573.pt'
#PBERT_WW = f'{base_dir}/PointLLM/checkpoint_point_encoder/Point-BERT.pth' #point_bert_v1.1.pt
PCs_PATH = '/media/data2/dmercanti/datasets/objaverse_660K/8192_npy/'
#TXT_EMB_PATH = '/media/data4/dmercanti/cap3D_txt_embeddings/'
DEV = "cuda:1"
BS = 128
K = int(sys.argv[1]) # 25


def main():
    
    id_to_texts_ = {}
    hard_dataset = defaultdict(lambda:[])
    device = torch.device(DEV if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    
    print(f'Loading texts from {ID_TEXT_FILE}')
    id_to_texts_ = remove_clouds_in(load_texts(ID_TEXT_FILE), f'{base_dir}/cross_coherence_objaverse/data/not_found')
    _, test_i2t = split(id_to_texts_, .95, .05, shuffle=deterministic_shuffle_dict, check_partition=True)
 
    print(f"Loading PointBERT; config from {PBERT_CONFIG_FILE}.")
    _, point_encoder = load_pt_encoder(PBERT_CONFIG_FILE, PBERT_WW,	f'{base_dir}/PointLLM/',
          device=device, use_max_pool=True)
    # use_max_pool=True to interpret the result as global without further proc.

    locator = lambda id_: f'{os.path.join(PCs_PATH, id_)}_8192.npy'
    sampler = get_sampler(test_i2t, locator, BS, device=device, deterministic=True, drop_last=True)
    
    ids = []
    embeddings = []
    texts = []
    for b, (bids, clouds, ts) in tqdm(enumerate(sampler),total=24448/(BS)):
        #if b==1:break #if b==(N//BS): break
        with torch.no_grad():
            pc_embs = point_encoder(clouds).to('cpu', copy=True, non_blocking=True)
            for i, pc_emb in enumerate(pc_embs):
                embeddings.append(pc_emb.view((768,)))
        ids += bids
        texts += ts
        del pc_embs, clouds
    N = b*BS
    print('embedding computation ended, N=', N)
    del point_encoder
    torch.cuda.empty_cache()
    
    gpu_embeddings = [_.to(device) for _ in embeddings]
    del embeddings
    print('tranfer to GPU ended, start of graph comp.')
    t0 = time.monotonic()
    
    graph = knn_graph(torch.stack(gpu_embeddings), k=K, cosine=True)
    for i in range(N):
        poss_inds = (graph[1] == i).nonzero(as_tuple=True)[0]
        graph_i = poss_inds[random.randrange(0,len(poss_inds))]
        #ind = poss_inds[graph_i].item()  # if duplicates exist, we choose one
        hard_dataset[i] = {'txt': texts[i], 'id': ids[i], 'knn-txt': texts[graph[0][graph_i]], 'knn-id': ids[graph[0][graph_i]]}
    print('graph computed in', round(time.monotonic()-t0))
    
    for i,k in enumerate(hard_dataset):
        print(k, hard_dataset[k])
        if i == 10:
            break
        
    print('Saving...')
    with open(f'./data/hard_{K}', 'w') as f:  #text-based
        #torch.save(dict(hard_dataset), f, pickle_protocol=0)
        f.write(str(dict(hard_dataset)))
        #f.seek(0)
        #print(torch.load(f))
    
    #break

    
if __name__ == '__main__':
    main()
    