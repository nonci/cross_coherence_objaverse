SHORT_TO_DEBUG = -1  # interrupt at nth. batch while train. and val.; -1 to disable
SHORTEN_VAL = False

import os, torch, sys, json
from tqdm import tqdm
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = '/home/dmercanti/dev'
DATA_DIR = '/media/data2/dmercanti/datasets'
LOG_DIR = f"{BASE_DIR}/cross_coherence_objaverse/exps/ignore"  # Folder to save training data and checkpoints

sys.path.append(f'{BASE_DIR}/shapeglot_text2shape/')
from custom_utils import permutation_indexes

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
DEV_TRAIN  = "cuda:1"  # Used for actual training
DEV_CLOUDS = "cuda:1"  # Used for cloud embeddings computation
MPOOL = False  # keep as it is!
B_SIZE = 32

# Training hyperparameters:
LR_0 = 5.e-4      # prove: 1e-4
LR_GAMMA = 0.97   # 0.97 in 02, 0.95 in 01
EPOCHS_BEFORE_DECAY = 5
TRAIN_EPOCHS = 3


def main():
    assert torch.cuda.is_available()
 
    print(f'Loading texts from \"{ID_TEXT_FILE.split("/")[-1]}\".')
    # keep existing clouds only:
    id_to_texts = remove_clouds_in(load_texts(ID_TEXT_FILE), f'{BASE_DIR}/cross_coherence_objaverse/data/not_found')
    print('NEW size (train+val+test) is:', len(id_to_texts))
    if SHORT_TO_DEBUG >0: print('WARNING: shortening to', SHORT_TO_DEBUG)
    
    # TEST SET is shuffled always in the same way:
    train_and_val_ds, test_ds = split(id_to_texts, .95, .05, shuffle=deterministic_shuffle_dict, check_partition=True)
    #print('These should be the same when splitting for testing:')
    #for n,k in enumerate(test_ds):
    #    if n==5: break
    #    print(' ', k)
    del test_ds
        
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
        device = DEV_TRAIN,
    ).to(DEV_TRAIN)
    listener.train()
    
    optimizer = optim.Adam(listener.parameters(), lr=LR_0)
    s1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1., total_iters=0)
    s2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA, verbose=True)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[s1, s2], milestones=[EPOCHS_BEFORE_DECAY])
    
    log_sigmoid = nn.LogSigmoid()
    
    #discarded = []  # file not found
    writer = SummaryWriter(LOG_DIR)
 
    for epoch in range(1, TRAIN_EPOCHS+1):
        train_i2t, val_i2t = split(train_and_val_ds, 1-5/95, 5/95, shuffle=deterministic_shuffle_dict, check_partition=True)
        train_size, val_size = len(train_i2t), len(val_i2t)
        
        locator = lambda id_: f'{os.path.join(PCs_PATH, id_)}_8192.npy'
        train_ds = get_sampler(train_i2t, locator, B_SIZE, device=DEV_CLOUDS,\
            deterministic=False, drop_last=True)
        val_ds_bs = get_sampler(val_i2t, locator, B_SIZE, device=DEV_CLOUDS,\
            deterministic=True, drop_last=True)
        val_ds_2 = get_sampler(val_i2t, locator, 2, device=DEV_CLOUDS,\
            deterministic=True, drop_last=True)
        
        n_examples = 0
        stats = Stats_handler(optimizer)
        listener.train()
        
        # TRAIN PHASE
        for b, (orig_ids, clouds, _) in tqdm(enumerate(train_ds), total=(train_size//B_SIZE) if SHORT_TO_DEBUG<0 else SHORT_TO_DEBUG): #
            if SHORT_TO_DEBUG==b: break
            
            n_examples += len(orig_ids)
            #if n_examples > 5000: break
            
            with torch.no_grad():   # encoders must not be trained
                enc_clouds = point_encoder(clouds) # [BSIZE, 513, 384]
                enc_clouds = enc_clouds.to(DEV_TRAIN)
                enc_texts = torch.stack(get_text_embeddings(orig_ids ,TXT_EMB_PATH, DEV_TRAIN, pad_to=None)) # list of [N_TOK_b, 1024], b<BSIZE OR [2, 77, 1024]
                # [n.tk, 75, 1024]
            
            repeated_txt_embeds = enc_texts.repeat(B_SIZE,1,1)
            clouds_shp = enc_clouds.shape[1:]
            repeated_cloud_embeds = enc_clouds.repeat(B_SIZE,1,1,1).permute(1,0,2,3).reshape(B_SIZE**2, *clouds_shp)
            # [BSIZE**2, 513, 384]
            
            # To avoid the net learning identity (batch-aware permutation):                        
            perm, new_col_labels, new_row_labels = permutation_indexes(B_SIZE, DEV_TRAIN)
            # We don't need to in-batch permute "repeated_clouds", given that it's always the same cloud in-batch.
            # train only:
            repeated_txt_embeds = repeated_txt_embeds[perm]  # [BSIZE**2, 75, 1024]
            #ids = [_  for _ in orig_ids for r in range(B_SIZE)]  # [BSIZE**2]
            
            logits = listener(
                repeated_cloud_embeds,
                repeated_txt_embeds,
            #    ids = ids
            ).view((B_SIZE, B_SIZE))  # row per each shape?
            
            logits = logits * torch.exp(listener.t) + listener.b
            # -1 matrix, with diagonal 1, permuted:
            labels = ((2 * torch.eye(B_SIZE, device=DEV_TRAIN) - torch.ones(B_SIZE, device=DEV_TRAIN)))
            # train only:
            labels = labels[new_row_labels]

            sigmoid_loss = -torch.sum(log_sigmoid(labels * logits)) / B_SIZE
            
            # Some useful stats:
            stats.update_stats(logits, new_col_labels, new_row_labels, sigmoid_loss.item()*B_SIZE) # CHECK
            
            # backward + optimize:
            listener.zero_grad()
            sigmoid_loss.backward()
            optimizer.step()
        
        lr, epoch_loss, epoch_acc, epoch_txt_acc, epoch_shp_acc = stats.get_and_reset_stats(n_examples)
        for what in ('lr', 'epoch_loss', 'epoch_acc', 'epoch_txt_acc', 'epoch_shp_acc'):
            writer.add_scalar(f'train/{what}', eval(what), epoch)
        writer.flush()
        
        lr_scheduler.step() # update lr at every epoch ; pass loss if needed
        
        # VAL. PHASE(s):
        #listener.load_state_dict(torch.load(\
        #    '/home/dmercanti/dev/cross_coherence_objaverse/exps/01/checkpoint_23.pth')['model_state'])
        
        for bs, val_ds in ( (B_SIZE, val_ds_bs), ):  # (2, val_ds_2),(B_SIZE, val_ds_bs) -> 2 validations are performed
            n_examples = 0
            stats = Stats_handler() # reset anyway
            listener.eval()
            t_range = torch.tensor(range(bs)).to(DEV_TRAIN)
            
            for b, (orig_ids, clouds, _) in tqdm(enumerate(val_ds), total=(val_size//bs)): #
                if SHORTEN_VAL and SHORT_TO_DEBUG==b: break
                n_examples += len(orig_ids)
                
                with torch.no_grad(): 
                    enc_clouds = point_encoder(clouds) # [BSIZE, 513, 384]
                    enc_clouds = enc_clouds.to(DEV_TRAIN)
                    enc_texts = torch.stack(get_text_embeddings(orig_ids ,TXT_EMB_PATH, DEV_TRAIN, pad_to=None)) # list of [N_TOK_b, 1024], b<BSIZE OR [2, 77, 1024]
                    # [n.tk, 75, 1024]
                
                    repeated_txt_embeds = enc_texts.repeat(bs,1,1)
                    clouds_shp = enc_clouds.shape[1:]
                    repeated_cloud_embeds = enc_clouds.repeat(bs,1,1,1).permute(1,0,2,3).reshape(bs**2, *clouds_shp)
                    # [BSIZE**2, 513, 384]
                    
                    # To avoid the net learning identity (batch-aware permutation):                        
                    #perm, new_col_labels, new_row_labels = permutation_indexes(bs, DEV_TRAIN)
                    # We don't need to in-batch permute "repeated_clouds", given that it's always the same cloud in-batch.
                    # train only:
                    #repeated_txt_embeds = repeated_txt_embeds[perm]  # [BSIZE**2, 75, 1024]
                    #ids = [_  for _ in orig_ids for r in range(bs)]  # [BSIZE**2]
                    
                    logits = listener(
                        repeated_cloud_embeds,
                        repeated_txt_embeds,
                        #ids = ids
                    ).view((bs, bs))  # row per each shape?
                    
                    logits = logits * torch.exp(listener.t) + listener.b
                    # -1 matrix, with diagonal 1, permuted:
                    labels = ((2 * torch.eye(bs, device=DEV_TRAIN) - torch.ones(bs, device=DEV_TRAIN)))
                    # train only:
                    #labels = labels[new_row_labels]
                    sigmoid_loss = -torch.sum(log_sigmoid(labels * logits)) / bs
                
                    # Some useful stats:
                    stats.update_stats(logits, t_range, t_range, sigmoid_loss.item()*bs)
                
                # backward + optimize:
                #listener.zero_grad()
                #sigmoid_loss.backward()
                #optimizer.step()
            
            _, epoch_loss, epoch_acc, epoch_txt_acc, epoch_shp_acc = stats.get_and_reset_stats(n_examples)
            #for what in ('epoch_loss', 'epoch_acc', 'epoch_txt_acc', 'epoch_shp_acc'):
            #    writer.add_scalar(f'val{bs}^2/{what}', eval(what), epoch)
            #writer.flush()
        
        if epoch_acc > 0.9:
            print(f'Saving checkpoint in {LOG_DIR} due to VAL. ACC. > 0.9')
            save_dict = {
                'epoch': epoch,
                'model_state': listener.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }
            torch.save(save_dict, f'{LOG_DIR}/checkpoint_{epoch}.pth')
        
            

'''		
    try:
        with open('not_found', 'w') as fp:
            json.dump([_.split('/')[-1] for _ in discarded], fp)
    except Exception:
        print(discarded[:10])
'''

    
if __name__ == '__main__':
    main()