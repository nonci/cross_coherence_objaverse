SHORT_TO_DEBUG = 5  # interrupt at nth. batch ; -1 to disable
MANUAL_TEST = True

import torch, time,sys
from tqdm import tqdm
from models.listener import Attention_Listener_v2

from objaverse_utils import load_texts, get_sampler_hard_ds, load_pt_encoder,\
    split, deterministic_shuffle_dict, remove_clouds_in
from models.mlp_decoder import MLPDecoder

BASE_DIR = '/home/dmercanti/dev'
DATA_DIR = '/media/data2/dmercanti/datasets'
CHECKPOINT = f"{BASE_DIR}/cross_coherence_objaverse/exps/01/checkpoint_23.pth"  # checkpoint to be loaded

# weights, see: https://github.com/OpenRobotLab/PointLLM/issues/1
#ID_TEXT_FILE = '/media/data2/dmercanti/datasets/objaverse_660K/Cap3D_automated_Objaverse.csv'  # 661,577 samples
ID_TEXT_FILE = f'{DATA_DIR}/objaverse_660K/Cap3D_automated_Objaverse_highquality.csv'  # 549,922 sAMPLES
PBERT_CONFIG_FILE = f'{BASE_DIR}/PointLLM/checkpoint_point_encoder/PointTransformer_tiny_8192point.yaml'  # default
PBERT_WW = f'{BASE_DIR}/PointLLM/checkpoint_point_encoder/pointBert_ww_Tiny/checkpoint_7573.pt'
PCs_PATH = f'{DATA_DIR}/objaverse_660K/8192_npy/'
TXT_EMB_PATH = '/media/data4/dmercanti/cap3D_txt_embeddings/'
USE_STORED_TEXTS = True
DEV_TEST  = "cuda:1"  # Used for actual test
DEV_CLOUDS ="cuda:1"  # Used for cloud embeddings computation
MPOOL = False  # keep as it is!
B_SIZE = 2
K = 1 #int(sys.argv[1]) #25
DATASET = f'./data/hard_{K}'


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
    
    print('Loading test metadata...')
    with open(DATASET, 'r') as f:
        test_metadata = eval(f.read())
    
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
 
    # test. PHASE(s):
    n_examples = 0
    ok = 0
    
    test_ds = get_sampler_hard_ds(
        test_metadata,
        PCs_PATH,
        TXT_EMB_PATH,
        DEV_TEST
    )
    
    for b, (txts, ids, clouds, txt_es, gt) in enumerate(test_ds): #tqdm , total=(test_size):
        if SHORT_TO_DEBUG==b: break
        
        with torch.no_grad(): 
            enc_clouds = point_encoder(clouds) # [BSIZE, 513, 384]
            enc_clouds = enc_clouds.to(DEV_TEST)
            enc_text = txt_es[gt]   # list of [N_TOK_b, 1024], b<BSIZE OR [2, 77, 1024]
            # [n.tk, 75, 1024]
            #repeated_txt_embeds = enc_text.repeat(B_SIZE,1,1)
            
            logits0 = listener(
                enc_clouds[0:1],
                txt_es[gt:gt+1]  #repeated_txt_embeds,
            )  # ex: tensor([[0.4599, 0.2646]], device='cuda:1')
            logits1 = listener(
                enc_clouds[1:2],
                txt_es[gt:gt+1]  #repeated_txt_embeds,
            )
            
            is_ok = (torch.max(torch.cat((logits0, logits1)), dim=0).indices[0] == gt).item()
            if MANUAL_TEST and not is_ok : #and abs(logits1-logits0)>=1.:
                print(txts[gt])
                print('ids:', ids)
                print('gt:', gt)
                print('CCs are:', logits0.item(), logits1.item())
                input()
            
            ok += is_ok
            n_examples += 1 # couples counter
            
        if n_examples%100==0:
            print('\r', ok, n_examples, round(ok/n_examples*100, 2), end=' '*20)
            
    print(ok, n_examples, round(ok/n_examples*100, 3), end=' '*20)
    with open(f'./data/hard_test_results', 'a') as f:
        f.write(f'K={K}; OK: {ok}; N_EX: {n_examples}; ACCURACY (rounded): {round(ok/n_examples*100, 3)}%; time: {time.ctime(time.time())}\n')

    
if __name__ == '__main__':
    main()
    
    
'''
19234 24400 78.83 on kNN k=1
22487 24400 92.16 on kNN k=50
'''


'''
K=1
0 {'txt': '3D rendering of a wall-mounted metal mailbox with a hole in the door.', 'id': '91a7341d39aa4e69b200f59dbec4df87', 'knn-txt': 'A 3D model featuring a metal box, white building, trash can, small box on dirt, toilet, broken concrete, and a street with a bench.', 'knn-id': '1babe5c25ea0426991d0bbfd6a3761af'}
1 {'txt': 'A 3D model of a snow globe featuring trees.', 'id': '494788a17a7e4c6d9ea62b43e2730607', 'knn-txt': 'A 3D model of a blue sphere.', 'knn-id': 'c475323dc7f24e26ba2009c08c8e1941'}
2 {'txt': '3D model of a medieval wooden table with a metal frame and various items on it.', 'id': 'fcf55b71a44a4903ab7bd0acdb49e1e2', 'knn-txt': 'A 3D model of a table with books, bottles, and various items, featuring a pink and red cloth.', 'knn-id': '57450896404c4450968189a7b0a8befe'}
3 {'txt': '3D model of a birthday cake with a candle.', 'id': '20d612c179a44c0c84e0ad13decc5cd0', 'knn-txt': '3D model of a strawberry cake with chocolate icing, a handbag, and strawberries on it.', 'knn-id': '9b0bd81f3d9744a384549d4895ec89f9'}
4 {'txt': 'A 3D model of a black, pink, and purple bouquet of flowers in a vase.', 'id': 'c8786842d545420996a4ad33f5137fec', 'knn-txt': '3D model of a tall grass plant with white flowers.', 'knn-id': '7561b4d61fcf4600b6ac6478fde78c01'}
5 {'txt': 'A 3D rendering of a staircase with a zebra-striped rug, glass top, and white ceiling light.', 'id': '0b884deaa7604320888d0843a237ce47', 'knn-txt': 'A 3D model of a white cube featuring a staircase with a black and white striped floor and glass elements.', 'knn-id': 'e81fb0b9a57d4994ad59d8170e08109a'}
6 {'txt': 'A 3D model of a small house with a red roof and balconies.', 'id': '6083bfbf3d9f41179c265a521331c46c', 'knn-txt': 'A 3D model of a house with a red roof and a balcony.', 'knn-id': 'c5e0584535514be0b96fad71f86bcc53'}
7 {'txt': '3D model of a wooden baby crib with a canopy.', 'id': '943ee9ee9fa94da9b0b1c56cd7938581', 'knn-txt': 'A 3D model of a wooden baby crib with a blue canopy and a baby inside.', 'knn-id': 'c71a49e399734d57a01920e2280b08cb'}
8 {'txt': 'A 3D model of a wooden staircase featuring a red arrow and railing.', 'id': '3ec443f3641c4b80a1b4fdf9b362337b', 'knn-txt': '3D model of a spiral staircase with railing and ceiling light fixture.', 'knn-id': 'c7e40ceeec4b4caca784458417651c09'}
9 {'txt': '3D model of a white skull', 'id': '8eddf79774ea449cb1ab6827433ea7c5', 'knn-txt': 'A 3D model of a white human skull.', 'knn-id': '2f65547b08fa4fbb83b296b91903fb4b'}
10 {'txt': 'A 3D model of a red, horned creature with spikes, resembling a dragon and demon hybrid.', 'id': 'd39fc9fdb0054b6b93905bf5db005550', 'knn-txt': 'A 3D model of a pink and grey Pokemon resembling a dragon-crocodile hybrid.', 'knn-id': '743d45349ced4358be5f01252b89e4b9'}

K=25
0 {'txt': '3D rendering of a wall-mounted metal mailbox with a hole in the door.', 'id': '91a7341d39aa4e69b200f59dbec4df87', 'knn-txt': 'A 3D model of a lighthouse with a small fountain, situated in a pond or lake.', 'knn-id': '3a0f05ac72484496b53b9550a11de942'}
1 {'txt': 'A 3D model of a snow globe featuring trees.', 'id': '494788a17a7e4c6d9ea62b43e2730607', 'knn-txt': 'White 3D sphere', 'knn-id': '46a151d8e969418484c32cad47f6e1e6'}
2 {'txt': '3D model of a medieval wooden table with a metal frame and various items on it.', 'id': 'fcf55b71a44a4903ab7bd0acdb49e1e2', 'knn-txt': 'A 3D model of a table with books, bottles, and various items, featuring a pink and red cloth.', 'knn-id': '57450896404c4450968189a7b0a8befe'}
3 {'txt': '3D model of a birthday cake with a candle.', 'id': '20d612c179a44c0c84e0ad13decc5cd0', 'knn-txt': 'A 3D model featuring a pizza in a cardboard box, a cup of coffee, and a wooden bookshelf with a green cover.', 'knn-id': 'f3d0c128e2554489b1c1c3afab974283'}
4 {'txt': 'A 3D model of a black, pink, and purple bouquet of flowers in a vase.', 'id': 'c8786842d545420996a4ad33f5137fec', 'knn-txt': 'Pink jellyfish with long tentacles and white accents', 'knn-id': '9c72edc0766f476498b3685bf8ea6030'}
5 {'txt': 'A 3D rendering of a staircase with a zebra-striped rug, glass top, and white ceiling light.', 'id': '0b884deaa7604320888d0843a237ce47', 'knn-txt': 'A 3D rendering of a white cube with stairs and a hole.', 'knn-id': '2ad74f66618344859d35ee4a7b1b0eb1'}
6 {'txt': 'A 3D model of a small house with a red roof and balconies.', 'id': '6083bfbf3d9f41179c265a521331c46c', 'knn-txt': 'A 3D model of a house with a red, yellow, and white roof.', 'knn-id': '4addf7513c5845a8beb0c9b325b5c817'}
7 {'txt': '3D model of a wooden baby crib with a canopy.', 'id': '943ee9ee9fa94da9b0b1c56cd7938581', 'knn-txt': 'A 3D model of a small wooden hut-like structure with a roof and a cross on top.', 'knn-id': 'd438fe21af3d49dca756bd1d0284a972'}
8 {'txt': 'A 3D model of a wooden staircase featuring a red arrow and railing.', 'id': '3ec443f3641c4b80a1b4fdf9b362337b', 'knn-txt': '3D model of a wooden table with stairs, accompanied by a chair and a square ceiling light.', 'knn-id': 'bd611d8a053c4f9cb9d60715622205bb'}
9 {'txt': '3D model of a white skull', 'id': '8eddf79774ea449cb1ab6827433ea7c5', 'knn-txt': 'A 3D model of a human skull.', 'knn-id': '211ae7bc07e1416b9ba82689ef24b9c2'}
10 {'txt': 'A 3D model of a red, horned creature with spikes, resembling a dragon and demon hybrid.', 'id': 'd39fc9fdb0054b6b93905bf5db005550', 'knn-txt': 'A 3D model of a red and yellow fish with a crown and yellow eyes.', 'knn-id': 'fafc2fa22e1c400da9a32e6f9359fa53'}


K=50
0 {'txt': '3D rendering of a wall-mounted metal mailbox with a hole in the door.', 'id': '91a7341d39aa4e69b200f59dbec4df87', 'knn-txt': 'A 3D model featuring a gravestone, tombstone, pillar, and obelisk on a platform.', 'knn-id': 'fdad2d44bbf94e29961b15c66ad07e0a'}
1 {'txt': 'A 3D model of a snow globe featuring trees.', 'id': '494788a17a7e4c6d9ea62b43e2730607', 'knn-txt': 'A purple sphere with stars on it.', 'knn-id': '7f5ee225c8dd43a68c1aaef8534c7d34'}
2 {'txt': '3D model of a medieval wooden table with a metal frame and various items on it.', 'id': 'fcf55b71a44a4903ab7bd0acdb49e1e2', 'knn-txt': '3D model of a small bar/restaurant building with a beer sign, balcony, and two beer mugs.', 'knn-id': 'be2dffbbdbfa424ab29ffdd83cec2975'}
3 {'txt': '3D model of a birthday cake with a candle.', 'id': '20d612c179a44c0c84e0ad13decc5cd0', 'knn-txt': 'A 3D model of a gold candle holder with a face and candelabra design.', 'knn-id': '498f2a97ec974e3182c0049b3e83608e'}
4 {'txt': 'A 3D model of a black, pink, and purple bouquet of flowers in a vase.', 'id': 'c8786842d545420996a4ad33f5137fec', 'knn-txt': 'A 3D model of a pink and purple earring with a cord, featuring a white box, purple wires, and a plug attachment.', 'knn-id': 'f0c5c875161242a2977996cef5113dd6'}
5 {'txt': 'A 3D rendering of a staircase with a zebra-striped rug, glass top, and white ceiling light.', 'id': '0b884deaa7604320888d0843a237ce47', 'knn-txt': 'A 3D rendering of a brick wall featuring a window, staircase, and table with a white top.', 'knn-id': 'fb45840e107a475dbb7af3996ad90497'}
6 {'txt': 'A 3D model of a small house with a red roof and balconies.', 'id': '6083bfbf3d9f41179c265a521331c46c', 'knn-txt': 'A 3D model of a modern two-story house with balconies and a brown roof.', 'knn-id': 'accdf77f9a1f48978d3846df4d1032a5'}
7 {'txt': '3D model of a wooden baby crib with a canopy.', 'id': '943ee9ee9fa94da9b0b1c56cd7938581', 'knn-txt': '3D model of a colorful ice cream shop with a cone, cart, table, and chairs, featuring umbrellas.', 'knn-id': 'c0806cfaa4fd43b0871b1c788120d159'}
8 {'txt': 'A 3D model of a wooden staircase featuring a red arrow and railing.', 'id': '3ec443f3641c4b80a1b4fdf9b362337b', 'knn-txt': 'A 3D model of a small white cube with stairs, a checkered floor, and a large square light fixture.', 'knn-id': 'c240816904cc4b948e3ec82e382ed216'}
9 {'txt': '3D model of a white skull', 'id': '8eddf79774ea449cb1ab6827433ea7c5', 'knn-txt': 'White 3D model of a human head with open mouth.', 'knn-id': '584b153c4d294086ae67a82c6d8e2344'}
10 {'txt': 'A 3D model of a red, horned creature with spikes, resembling a dragon and demon hybrid.', 'id': 'd39fc9fdb0054b6b93905bf5db005550', 'knn-txt': '3D model of an orange fish with spikes.', 'knn-id': '95f5513c8bdf407681a75507c1a6237c'}

'''