'''
This script tests CC by ranking N_TEXTS wrt. a shape and returning the GT text
position inside the rank (should be as low as possible).
Configure by editing cc_rank.ini and passing experiment name.
'''

import torch, time, sys
from tqdm import tqdm
from models.listener import Attention_Listener_v2

from objaverse_utils import load_texts, get_sampler_hard_ds, load_pt_encoder,\
    split, deterministic_shuffle_dict, remove_clouds_in, avail_gpus, info, warn
from models.mlp_decoder import MLPDecoder
from custom_pprint import pprint
import configparser

TEST = sys.argv[1] if len(sys.argv)>1 else 'p.bert_CC_Cap3D'

SHORT_TO_DEBUG = -1  # interrupt at nth. cloud ; -1 to disable
# MANUAL_TEST = True

config_ = configparser.ConfigParser()
config_.read('cc_rank.ini')
config = config_[TEST] 


def ranking_string(ok, n_ex, time_, gt_positions):
    s = f'N={config["N"]}/K={config["K"]}; OK: {ok}; N_EX: {n_ex}; ACCURACY (rounded): {round(ok/n_ex*100, 3)}%\n'\
        f'End time: {time.ctime(time_)}\n'
    for i,n in enumerate(gt_positions):
        s = s + f'{i+1:3}) {n:>6}  ({round((n/n_ex*100).item(), 2):02}%){"  <-- correct ones" if not i else ""}\n'
    return s


def main():
    assert torch.cuda.is_available()
    
    DEV_TEST = config['DEV_TEST'] if config['DEV_TEST']!='auto' else f'cuda:{avail_gpus()[0]}'
    DEV_CLOUDS = config['DEV_CLOUDS'] if config['DEV_CLOUDS']!='auto' else f'cuda:{avail_gpus()[0]}'
    info('Using devices: ', DEV_TEST, DEV_CLOUDS)

    info(f'Loading texts from \"{config["ID_TEXT_FILE"].split("/")[-1]}\".')
    id_to_texts = remove_clouds_in(
        load_texts(config["ID_TEXT_FILE"]),
        config['NOT_FOUND_LIST']
    )
    info('NEW size (train+val+test) is:', len(id_to_texts))
    
    if SHORT_TO_DEBUG > 0:
        warn('Shortening to', SHORT_TO_DEBUG)
    
    # TEST SET is shuffled always in the same way, for reproducib.:
    _, test_i2t = split(id_to_texts, .95, .05, shuffle=deterministic_shuffle_dict, check_partition=True)
    info('Test size:', len(test_i2t))
    
    # Loading id and txt of knn(s)
    info('Loading test metadata from', config['DATASET'])
    with open(config['DATASET'], 'r') as f:
        test_metadata = eval(f.read())
    
    info(f"PointBERT config taken from:\n       {config['PBERT_CONFIG_FILE'].split('/')[-1]}")
    info(end = ' Loading PointBERT encoder... ')
    _, point_encoder = load_pt_encoder(
        config['PBERT_CONFIG_FILE'],
        config['PBERT_WW'],
        config['POINTLLM_CODE_DIR'],
        device=DEV_CLOUDS,
        use_max_pool=False
    )
    print('ok')
    
    mlp_decoder = MLPDecoder(1024, [100, 50, 1], use_b_norm=True, dropout=False)
    
    listener = Attention_Listener_v2(
        mlp_decoder,
        cloud_dim = 384,  # n. of dimensions, per feature
        text_dim = 1024,  # from T5!
        n_heads = 8,
        head_dim = 64,
        t0b0 = (40., -0.2),  # or use None, instead
        device = DEV_TEST,
    ).to(DEV_TEST)
    listener.eval()
    info('Loading AttentionListener checkpoint')
    listener.load_state_dict(torch.load(config['CHECKPOINT'])['model_state'])
    
    
    # test. PHASE:
    n_examples = 0
    ok = 0
    gt_positions = torch.zeros(int(config['N'])+1, dtype=int)
    
    test_ds = get_sampler_hard_ds(
        test_metadata,
        config['PCs_PATH'],
        config['TXT_EMB_PATH'],
        DEV_TEST,
        gt_cloud_only = True  # Only GT point cloud is needed for this test
    )
    
    for b, (txts, ids, cloud, txt_es, gt_index) in enumerate(test_ds): #tqdm , total=(test_size):
        if SHORT_TO_DEBUG == b:
            print()
            info('Stopping due to shortening set to', SHORT_TO_DEBUG)
            break
        
        with torch.no_grad():
            enc_cloud = point_encoder(cloud) # [1, 513, 384]
            enc_cloud = enc_cloud.to(DEV_TEST)
            #enc_text = txt_es[gt]   # list of [N_TOK_b, 1024], b<BSIZE OR [2, 77, 1024]
            # [n.tk, 75, 1024]
            #repeated_txt_embeds = enc_text.repeat(B_SIZE,1,1)
            
            logits = listener(
                enc_cloud.expand(len(ids), *enc_cloud.shape[1:]),  # shallow copy! Read only
                txt_es
            )  # ex: tensor([[0.4599, 0.2646]], device='cuda:1')
            
            is_ok = (torch.max(logits, dim=0).indices[0] == gt_index).item()
            # More specific:
            gt_position = ((logits.sort(dim=0, descending=True, stable=True).indices)==gt_index).\
                nonzero(as_tuple=True)[0].item()
            gt_positions[gt_position] += 1
            
            # if MANUAL_TEST and is_ok:  # not is_ok
            #     #cc_err = abs(logits1-logits0)
            #     #if 0.5<cc_err<1:
            #     print(txts[gt])
            #     print('ids:', ids)
            #     print('gt:', gt)
            #     print('CCs are:', logits0.item(), logits1.item(), '\n')
            #     print()
            
            ok += is_ok
            n_examples += 1 # couples counter
            
        if n_examples%100==0:
            print('\r', ok, n_examples, round(ok/n_examples*100, 2), end=' '*20)
    
    test_ends_at = time.time()
    
    print('\n' + ranking_string(ok, n_examples, test_ends_at, gt_positions))

    if config_.getboolean(TEST, 'WRITE_RESULTS'):
        with open(f'./data/hard_test_results_new', 'a') as f:
            f.write(ranking_string(ok, n_examples, test_ends_at, gt_positions)+'\n')
        info('Results written on ./data/hard_test_results_new')


if __name__ == '__main__':
    main()