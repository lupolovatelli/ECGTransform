import torch
import torch.nn as nn
import torch.nn.functional as F

# source /Users/Lovatelli/Desktop/3_year_project/MIMIC_DATA/Models/MI_NET_MIMIC/venv/bin/activate


# ===================
# ECG tokenizer
# ===================
def tokenizer(X,token_length=100,stride=50):
    """
    Efficient tokenization using torch.unfold
    Input: X [N, 12, L]
    Output: [N, 12, n_tokens, token_length]
    """
    X_tokens = X.unfold(dimension=2,size=token_length,step=stride)
    # X.unfold → [N, 12, n_tokens, token_length]
    return X_tokens.contiguous()



# ===================
# Sequence temporal encoder
# ===================
# def time_seq_encoding(X_tokens):
#     """
#     Time wise sequence encoding
#     Performed by creating an offset based on tokens's temporal delay  
#     Input: X_tokens [N,12,99,L]
#     Output: X_tokens_t_enc [N,12,99,L]
#     """

#     N,C,T,L = X_tokens.shape

#     token_positions = torch.linspace(
#         0,1,T,device=X_tokens.device
#     )

#     token_positions = token_positions.view(1,1,T,1)

#     return X_tokens + token_positions

def time_seq_encoding(x):
    """
    Input: [B, T, E]
    Output: [B, T, E]
    """
    B, T, E = x.shape

    token_positions = torch.linspace(
        0, 1, T, device=x.device
    ).view(1, T, 1)

    return x + token_positions

# ===================
# Spatial positional encoder
# ===================
def spatial_seq_encoding(X_tokens):
    """
    Time wise sequence encoding
    Performed by creating an offset based on tokens's temporal delay  
    Input: X_tokens [N,12,emb_dim]
    Output: X_tokens_t_enc [N,12,emb_dim]
    """

    N,C,E = X_tokens.shape

    token_positions = torch.linspace(
        0,1,C,device=X_tokens.device
    )

    token_positions = token_positions.view(1,C,1)

    return X_tokens + token_positions

    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model=100,num_heads=2):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim =  d_model//num_heads

        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)

    def forward(self,x):
        B, num_tokens, d_model = x.shape

        # [B,num_tokens, num_heads=5,head_dim=100/5]
        Q = self.W_q(x).view(B, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q,K.transpose(-2,-1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores,dim=-1)
        out = torch.matmul(attn,V)

        out = out.transpose(1, 2).contiguous().view(B, num_tokens, d_model)
        return self.W_o(out)


class TransBlock(nn.Module):
    def __init__(self,d_model=100,MHA_heads=2,dropout=0.1):
        """
        Temporal Transformer Encoder block  
        Architecture:
            Block: 
                - Normalization (layer norm)
                - Mult-head attention
                - Normalization (layer norm?)
                - Linear layer (hyperparam: 100,dropout)

        Input: Batch of 1 lead: [B,99,100] (already flattened from [B,1,99,100] if necessary)
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads = MHA_heads,
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model,d_model),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        # x: [B, T, 100] -> [B,99,100]
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x



class TransEncoder(nn.Module):
    """
    Temporal Transformer Encoder   
    Architecture:
        - Linear layer to make MHA possible with FOV = subset of feature-space 
        - TransTimeBlock x 2

    Input: Batch of 1 lead: [B,99,100] (already flattened from [B,1,99,100] if necessary)
    """
    def __init__(self,input_dim=100,d_model=100,MHA_heads=2,dropout=0.1,num_layers=2,temporal= True):
        super().__init__()

        self.input_proj = nn.Linear(input_dim,d_model)
        self.MHA_blocks = nn.Sequential(
            *[TransBlock(d_model=d_model, MHA_heads=MHA_heads, dropout=dropout) 
              for _ in range(num_layers)]
        )

        if temporal:
            self.pos_encoding = time_seq_encoding
        else:
            self.pos_encoding = spatial_seq_encoding

    def forward(self,x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.MHA_blocks(x)
        return x


class MultiLeadTemporalEncoder(nn.Module):
    def __init__(self, num_tokens=99,input_dim=100,d_model=100,hidden_dim=10,
                 MHA_heads=2,dropout=0.1,num_layers=2,num_leads=12):
        
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = TransEncoder(input_dim=input_dim,
                                                d_model=d_model,
                                                MHA_heads=MHA_heads,
                                                dropout=dropout,
                                                num_layers=num_layers)
        
        # Shared per-lead FC
        self.stack_fc = nn.Linear(d_model,hidden_dim)
        self.dim_reduc = nn.Linear(num_tokens*hidden_dim,128)

    def forward(self,x):
        B, C, T, F = x.shape
        # Flatten batch + leads to process independently
        x = x.view(B*C, T, F)          # [B*12, T, F]
        x = self.encoder(x)            # [B*12, T, d_model]
        x = self.stack_fc(x)
        x = x.view(B, C, T, -1)        # [B, 12, T, d_model]
        x = x.view(B,C,T*self.hidden_dim)
        x = self.dim_reduc(x)          # Outputs: [B,12,128] a 128 embedding per lead token ready for the spacial encoder should i retokenize?
        return x
    

class LeadSpatialEncoder(nn.Module):
    def __init__(self, input_dim=128, d_model=128, MHA_heads=2,
                 dropout=0.1,num_layers=2):
        
        super().__init__()
        self.temporal = False
        self.encoder = TransEncoder(input_dim=input_dim,
                                                d_model=d_model,
                                                MHA_heads=MHA_heads,
                                                dropout=dropout,
                                                num_layers=num_layers,
                                                temporal=self.temporal)

    def forward(self,x):
        x = self.encoder(x)
        return x


class MLP_Decoder(nn.Module):
    def __init__(self,emb_dim=12*128,dropout=0.1,num_pos_classes=1):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_dim,emb_dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim//2,num_pos_classes)
        )


    def forward(self,x):
        B,T,E = x.shape
        x = x.view(B,T*E)
        x = self.decoder(x)
        return x
    

class ECGTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.name = "ECGTransform"

        self.temporal_encoder = MultiLeadTemporalEncoder()
        self.spatial_encoder = LeadSpatialEncoder()
        self.decoder = MLP_Decoder()

    def forward(self,x):

        x = tokenizer(x)                # [B,12,T,100]

        x = self.temporal_encoder(x)    # [B,12,128]

        x = self.spatial_encoder(x)     # [B,12,128]

        x = self.decoder(x)             # [B,1]

        return x
