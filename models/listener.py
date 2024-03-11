'''
...
'''

import torch
from torch import nn
from .attention import CrossAttention


class Attention_Listener_v2(nn.Module):
    def __init__(self, mlp_decoder, cloud_dim, text_dim, n_heads, head_dim, \
        		dropout=0.0, t0b0=(10., 0.), device='cpu'):
        ''' This version requires both texts and clouds passed to fwd already in embedding space.
        
        Parameters:
        	mlp_decoder: the model to be used as final block
        	cloud_dim: shape of the clouds emb. to be fed to layernorm and attention
        	text_dim:  shape of the text emb. to be fed to layernorm and attention
        	n_heads: number of attention heads
        	head_dim: dimension of each head
        	dropout: dropout rate to be applied to CrossAttention
        	t0b0: if not None, t and b are initialized as parameters for Sigmoid loss
        	device: torch device, defaults to "cpu".  '''
         
        super(Attention_Listener_v2, self).__init__()
        
        self.logit_encoder = mlp_decoder
        #self.proj_feats = nn.Linear(context_dim, n_heads*d_head, bias=False)   # projection layer for global embeddings
        # context_dim = text embedding dim
        
        self.attn1 = CrossAttention(query_dim=cloud_dim, context_dim=text_dim, heads=n_heads, \
            						dim_head=head_dim, inner_dim=n_heads*head_dim, dropout=dropout)
        
        self.attn2 = CrossAttention(query_dim=text_dim, context_dim=cloud_dim, heads=n_heads, \
            						dim_head=head_dim, inner_dim=n_heads*head_dim, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(cloud_dim)
        self.norm2 = nn.LayerNorm(text_dim)
        self.device=device
        if t0b0 is not None:
            t0, b0 = torch.log10(torch.tensor(t0b0[0])), torch.tensor(t0b0[1])
            self.t = torch.nn.parameter.Parameter(data=t0, requires_grad=True)
            self.b = torch.nn.parameter.Parameter(data=b0, requires_grad=True)
        
        
    def forward(self, pc_feats, text_embed, *_):
        mask = text_embed!=0    # (B, seq_len, 1024)
        mask = mask[:,:,0]      # (B, seq_len): all the 1024 tensors have the same True and False values  

        x_1 = self.attn1(self.norm1(pc_feats), context=text_embed, mask=mask)       # x_1: Bx128x512
        x_2 = self.attn2(self.norm2(text_embed), context=pc_feats)                  # x_2: Bx42x512

        mean_x1 = torch.mean(x_1, dim=1)  # [B, 513, 512]
        mean_x2 = torch.mean(x_2, dim=1)  # [B, 75, 512]
        feats = torch.cat((mean_x1, mean_x2), dim=1)  # [B, 1024]

        ef = self.logit_encoder(feats)        # [B, 1]
        return ef
        