SHORT_TO_DEBUG = -1  # interrupt at nth. batch ; -1 to disable

import os, torch, sys, json
from tqdm import tqdm
from torch import optim, nn

BASE_DIR = '/home/dmercanti/dev'
DATA_DIR = '/media/data2/dmercanti/datasets'
CHECKPOINT = f"{BASE_DIR}/cross_coherence_objaverse/exps/01/checkpoint_14.pth"  # checkpoint to be loaded

#sys.path.append(f'{BASE_DIR}/shapeglot_text2shape/')
#from custom_utils import permutation_indexes

from models.listener import Attention_Listener_v2

from objaverse_utils import load_texts, get_text_embeddings, get_sampler, load_pt_encoder,\
    split, deterministic_shuffle_dict, Stats_handler, remove_clouds_in
from models.mlp_decoder import MLPDecoder

# weights, see: https://github.com/OpenRobotLab/PointLLM/issues/1
#ID_TEXT_FILE = '/media/data2/dmercanti/datasets/objaverse_660K/Cap3D_automated_Objaverse.csv'  # 661,577 samples
ID_TEXT_FILE = f'{DATA_DIR}/objaverse_660K/Cap3D_automated_Objaverse_highquality.csv'  # 549,922 sAMPLES
PBERT_CONFIG_FILE = f'{BASE_DIR}/PointLLM/checkpoint_point_encoder/PointTransformer_tiny_8192point.yaml'  # default
PBERT_WW = f'{BASE_DIR}/PointLLM/checkpoint_point_encoder/pointBert_ww_Tiny/checkpoint_7573.pt'
PCs_PATH = f'{DATA_DIR}/objaverse_660K/8192_npy/'
TXT_EMB_PATH = '/media/data4/dmercanti/cap3D_txt_embeddings/'
USE_STORED_TEXTS = True
DEV_TEST  = "cpu"  # Used for actual test
DEV_CLOUDS ="cpu"  # Used for cloud embeddings computation
MPOOL = False  # keep as it is!
B_SIZE = 2


def main():
    assert torch.cuda.is_available()
 
    print(f'Loading texts from \"{ID_TEXT_FILE.split("/")[-1]}\".')
    id_to_texts = remove_clouds_in(load_texts(ID_TEXT_FILE), f'{BASE_DIR}/cross_coherence_objaverse/data/not_found')
    print('NEW size (train+val+test) is:', len(id_to_texts))
    if SHORT_TO_DEBUG >0: print('WARNING: shortening to', SHORT_TO_DEBUG)
    
    # TEST SET is shuffled always in the same way:
    _, test_i2t = split(id_to_texts, .95, .05, shuffle=deterministic_shuffle_dict, check_partition=True)
    #for n,k in enumerate(test_i2t):
    #    if n==5: break
    #    print(' ', k)
        
    print(f"PointBERT config taken from \"{PBERT_CONFIG_FILE.split('/')[-1]}\".")
    print(end = 'Loading PointBERT encoder... ')
    _, point_encoder = load_pt_encoder(
        PBERT_CONFIG_FILE,
        PBERT_WW,
        f'{BASE_DIR}/PointLLM/',
        device=DEV_CLOUDS,
        use_max_pool=MPOOL
    )
    print('ok')
    
    mlp_decoder = MLPDecoder(1024, [100, 50, 1], use_b_norm=True, dropout=False)
    
    listener = Attention_Listener_v2(
        mlp_decoder,
        cloud_dim = 384,  # n. of dimensions, per feature
        text_dim = 1024, # from T5!
        n_heads = 8,
        head_dim = 64,
        t0b0 = (40., -0.2),  # or use None, instead
        device = DEV_TEST,
    ).to(DEV_TEST)
    listener.eval()
    listener.load_state_dict(torch.load(CHECKPOINT)['model_state'])
    
    log_sigmoid = nn.LogSigmoid()
    
    #discarded = []  # file not found
    #writer = SummaryWriter(LOG_DIR)
 
    # test. PHASE(s):
    n_examples = 0
    stats = Stats_handler()
    t_range = torch.tensor(range(B_SIZE)).to(DEV_TEST)
    
    
    test_size = len(test_i2t)
    locator = lambda id_: f'{os.path.join(PCs_PATH, id_)}_8192.npy'
    test_ds = get_sampler(test_i2t, locator, B_SIZE, device=DEV_CLOUDS, deterministic=True, drop_last=True)
    
    for b, (orig_ids, clouds, _) in tqdm(enumerate(test_ds), total=(test_size//B_SIZE)):
        if SHORT_TO_DEBUG==b: break
        n_examples += len(orig_ids)
        
        with torch.no_grad(): 
            enc_clouds = point_encoder(clouds) # [BSIZE, 513, 384]
            enc_clouds = enc_clouds.to(DEV_TEST)
            enc_texts = torch.stack(get_text_embeddings(orig_ids, TXT_EMB_PATH, DEV_TEST, pad_to=None)) # list of [N_TOK_b, 1024], b<BSIZE OR [2, 77, 1024]
            # [n.tk, 75, 1024]
        
            repeated_txt_embeds = enc_texts.repeat(B_SIZE,1,1)
            clouds_shp = enc_clouds.shape[1:]
            repeated_cloud_embeds = enc_clouds.repeat(B_SIZE,1,1,1).permute(1,0,2,3).reshape(B_SIZE**2, *clouds_shp)
            
            #ids = [_  for _ in orig_ids for r in range(B_SIZE)]  # [BSIZE**2]
            
            logits = listener(
                repeated_cloud_embeds,
                repeated_txt_embeds,
                #ids = ids
            ).view((B_SIZE, B_SIZE))  # row per each shape?
            
            logits = logits * torch.exp(listener.t) + listener.b
            # -1 matrix, with diagonal 1:
            labels = ((2 * torch.eye(B_SIZE, device=DEV_TEST) - torch.ones(B_SIZE, device=DEV_TEST)))
            sigmoid_loss = -torch.sum(log_sigmoid(labels * logits)) / B_SIZE
        
            # Some useful stats:
            stats.update_stats(logits, t_range, t_range, sigmoid_loss.item()*B_SIZE)
    
    _, epoch_loss, epoch_acc, epoch_txt_acc, epoch_shp_acc = stats.get_and_reset_stats(n_examples)
    for what in ('epoch_loss', 'epoch_acc', 'epoch_txt_acc', 'epoch_shp_acc'):
        print(f'test b.size={B_SIZE}^2 - {what:20}', eval(what).item() if type(eval(what)) is torch.Tensor else eval(what))

    
if __name__ == '__main__':
    main()
    
    
'''
rel.acc:  tensor(0.9980, device='cuda:3', dtype=torch.float64) - abs.acc.: tensor(24642., device='cuda:3', dtype=torch.float64)
test b.size=2^2 - epoch_loss           0.22219241070915255
test b.size=2^2 - epoch_acc            0.9979750526486311
test b.size=2^2 - epoch_txt_acc        0.9989470273772881
test b.size=2^2 - epoch_shp_acc        0.9989875263243155
'''