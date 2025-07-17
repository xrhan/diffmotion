import torch
from video_model.u_vit3d_blocks import *
from video_model.modules3d import RotaryEmbedding3D
from functools import reduce


try:
    import natten
except ImportError:
    natten = None


class NormalizeWithCond2D(nn.Module):
    """
    Conditioning block for U-ViT, that injects external conditions into the network using FiLM.
    """
    def __init__(self, dim: int, emb_dim: int):
        super().__init__()
        self.emb_layer = nn.Conv2d(emb_dim, dim * 2, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6, affine=True)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        scale, shift = self.emb_layer(emb).chunk(2, dim=1)
        return self.norm(x) * (1 + scale) + shift
    
    
class NeighborhoodSelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads, emb_dim, kernel_size, res, dropout=0.0):
        super().__init__()
        self.n_heads = heads
        self.d_heads = dim // heads
        self.kernel_size = kernel_size
        self.norm = NormalizeWithCond2D(dim, emb_dim)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.scale = self.d_heads ** -0.5
        # self.pos_emb = AxialRoPE(self.d_heads // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = zero_module(nn.Linear(dim, dim, bias=False))
        self.q_norm, self.k_norm = Normalize(self.d_heads), Normalize(self.d_heads)
        self.rope = RotaryEmbedding2D(dim // heads, (res, res), flatten=False)

    def forward(self, x, cond):
        B, C, H, W = x.shape
        skip = self.norm(x, cond) #TODO:FIX THIS
        skip_perm = skip.permute(0, 2, 3, 1)  # (B, H, W, C)
        
        x_flat = skip_perm.view(B, H * W, C)  # (B, H*W, C)
        qkv = self.qkv_proj(x_flat)       
        q, k, v = rearrange(qkv, "b (hw) (qkv h d) -> qkv b h hw d", qkv=3, h=self.n_heads)
        # Reshape the flattened tokens back to the spatial grid (H, W).
        q = q.view(B, self.n_heads, H, W, self.d_heads)
        k = k.view(B, self.n_heads, H, W, self.d_heads)
        v = v.view(B, self.n_heads, H, W, self.d_heads)
        # Normalize queries and keys along the feature dimension.
        q = self.q_norm(q)
        k = self.k_norm(k)
        # Apply the rotary embedding.
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        # Ensure that 'natten' is available for neighborhood attention.
        if natten is None:
            raise ModuleNotFoundError("natten is required for neighborhood attention")
        
        # Compute neighborhood attention: these functions expect q, k, v as 5D tensors.
        qk = natten.functional.na2d_qk(q, k, self.kernel_size)
        qk = qk * self.scale
        a = torch.softmax(qk, dim=-1)
        x_attn = natten.functional.na2d_av(a, v, self.kernel_size)
        # x is now (B, n_heads, H, W, d_heads)
        # Rearrange back to (B, H, W, C) by concatenating the head dimensions.
        x_attn = rearrange(x_attn, "b h H W d -> b H W (h d)")
        
        x_attn = self.dropout(x_attn)
        x_proj = self.out_proj(x_attn)
        # Add the residual (skip) connection.
        x_out = x_proj + skip_perm  
        x_out = x_out.permute(0, 3, 1, 2)
        return x_out
    

# ------------------ 2D NATTEN (FIXED NORMALIZATION POSITION) ------------------
class NeighborhoodSelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads, emb_dim, kernel_size, res, attn_dropout=0.0):
        super().__init__()
        self.n_heads = heads
        self.d_heads = dim // heads
        self.kernel_size = kernel_size
        self.norm = NormalizeWithCond2D(dim, emb_dim)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.scale = self.d_heads ** -0.5
        # self.pos_emb = AxialRoPE(self.d_heads // 2, self.n_heads)
        self.dropout = nn.Dropout(attn_dropout)
        self.out_proj = zero_module(nn.Linear(dim, dim, bias=False))
        self.q_norm, self.k_norm = Normalize(self.d_heads), Normalize(self.d_heads)
        self.rope = RotaryEmbedding2D(dim // heads, (res, res), flatten=False)

    def forward(self, x, cond):
        B, C, H, W = x.shape
        _x = x # skip term
        x = self.norm(x, cond) #TODO:FIX THIS
        x_perm = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        
        x_flat = x_perm.view(B, H * W, C)  # (B, H*W, C)
        qkv = self.qkv_proj(x_flat)       
        q, k, v = rearrange(qkv, "b (hw) (qkv h d) -> qkv b h hw d", qkv=3, h=self.n_heads)
        # Reshape the flattened tokens back to the spatial grid (H, W).
        q = q.view(B, self.n_heads, H, W, self.d_heads)
        k = k.view(B, self.n_heads, H, W, self.d_heads)
        v = v.view(B, self.n_heads, H, W, self.d_heads)
        # Normalize queries and keys along the feature dimension.
        q = self.q_norm(q)
        k = self.k_norm(k)
        # Apply the rotary embedding.
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        # Ensure that 'natten' is available for neighborhood attention.
        if natten is None:
            raise ModuleNotFoundError("natten is required for neighborhood attention")
        
        # Compute neighborhood attention: these functions expect q, k, v as 5D tensors.
        qk = natten.functional.na2d_qk(q, k, self.kernel_size)
        qk = qk * self.scale
        a = torch.softmax(qk, dim=-1)
        a = self.dropout(a) # fix dropout location at qk score
        x_attn = natten.functional.na2d_av(a, v, self.kernel_size)

        x_attn = rearrange(x_attn, "b h H W d -> b H W (h d)")        
        x_proj = self.out_proj(x_attn)
        # Add the residual (skip) connection.
        x_out = x_proj.permute(0, 3, 1, 2)
        return x_out + _x
    

class NeighborhoodTransformerLayer(nn.Module): # similar to ResBlock style
    def __init__(self, dim, heads, emb_dim, kernel_size, res, dropout=0.0):
        super().__init__()
        self.self_attn = NeighborhoodSelfAttentionBlock(dim, heads, emb_dim, kernel_size, res, attn_dropout=dropout)
        self.emb_layer = nn.Conv2d(emb_dim, dim * 2, kernel_size=(1, 1))
        self.out_norm = NormalizeWithBias(dim)
        self.out_rest = nn.Sequential(
            nn.SiLU(),
            zero_module(
                nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1))
            ),
        )
    
    def _rearrange_noise_emb_nattn(self, x: Tensor, emb: Tensor):
        b_t, c, h, w = x.shape # no need to reshape x
        emb = repeat(emb, "b_t c -> b_t c h w", h=h, w=w)
        return emb

    def forward(self, x, emb):
        emb = self._rearrange_noise_emb_nattn(x, emb)
        h = self.self_attn(x, emb)
        emb_out = self.emb_layer(emb if emb.dim() == 4 else emb[:, :, None, None])
        scale, shift = emb_out.chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_rest(h)
        return x + h


# ------------------ 3D NATTEN ---------------------
# following https://github.com/SHI-Labs/NATTEN/blob/ea9ebe366d951984609b42f8cd8712d030a72d4a/src/natten/na3d.py#L89
class Neighborhood3DAttentionLayer_TStyle(nn.Module):
    '''
    Local 3D Neighborhood Attention with Transformer Style MLP Out Layer
    '''
    def __init__(self, dim, heads, emb_dim, kernel_size, res, temporal_length=3, attn_dropout=0.0, proj_dropout = 0.0):
        super().__init__()
        self.n_heads = heads
        self.dim_head = dim // heads
        self.kernel_size = kernel_size

        self.scale = self.dim_head ** -0.5
        self.temporal_length = temporal_length
        
        # self.pos_emb = AxialRoPE(self.d_heads // 2, self.n_heads)
        self.attn_drop_rate = attn_dropout
        self.attn_dropout = nn.Dropout(self.attn_drop_rate)

        self.attn_out = zero_module(nn.Linear(dim, dim, bias=False))
        self.norm = NormalizeWithCond(dim, emb_dim)
        self.q_norm, self.k_norm = Normalize(self.dim_head), Normalize(self.dim_head)
        
        mlp_dim = 4 * dim
        self.fused_dims = (3 * dim, mlp_dim)
        self.fused_attn_mlp_proj = nn.Linear(dim, sum(self.fused_dims), bias=True)
        self.rope = RotaryEmbedding3D(self.dim_head, (self.temporal_length, res, res),)

        self.mlp_out = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(proj_dropout),
            zero_module(nn.Linear(mlp_dim, dim, bias=True)),
        )
    
    def _rearrange_noise_emb_nattn(self, x: Tensor, emb: Tensor):
        b_t, c, h, w = x.shape # no need to reshape x
        x = rearrange(x, '(b t) c h w -> b t c h w', h=h, w=w, t=self.temporal_length)
        emb = repeat(emb, "(b t) c -> b (t h w) c", t=self.temporal_length, h=h, w=w)
        return x, emb
    
    def _unrearrange_3d_natten(self, x: Tensor, h, w) -> Tensor:
        x = rearrange(x, "b (t h w) c -> (b t) c h w", t=self.temporal_length, h=h, w=w)
        return x

    def forward(self, x, emb):
        x, emb = self._rearrange_noise_emb_nattn(x, emb)

        B, T, C, H, W = x.shape
        skip_term = x.permute(0, 1, 3, 4, 2)  # (B, T, H, W, C)
        _x = skip_term

        # ----------- 3D Neighborhood Attention -----------

        x_flat = skip_term.reshape(B, T * H * W, C) # (B, N, C)
        x = self.norm(x_flat, emb) # RMS Norm
        
        qkv, mlp_h = self.fused_attn_mlp_proj(x).split(self.fused_dims, dim=-1)
        qkv = rearrange(qkv, "b (thw) (qkv h d) -> qkv b h thw d", qkv=3, h=self.n_heads, d=self.dim_head)
        q, k, v = qkv.unbind(0) 
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)

        # Reshape for neighborhood attention: (b, t, h, w, heads, dim_head)
        q = rearrange(q, 'b nh (t h1 w1) d -> b nh t h1 w1 d', b=B, nh=self.n_heads, t=T, h1=H, w1=W)
        k = rearrange(k, 'b nh (t h1 w1) d -> b nh t h1 w1 d', b=B, nh=self.n_heads, t=T, h1=H, w1=W)
        v = rearrange(v, 'b nh (t h1 w1) d -> b nh t h1 w1 d', b=B, nh=self.n_heads, t=T, h1=H, w1=W)

        q = q * self.scale
        attn = natten.functional.na3d_qk(q, k, kernel_size=self.kernel_size, is_causal=False, dilation=1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        x_attn = natten.functional.na3d_av(attn, v, kernel_size=self.kernel_size, is_causal=False, dilation=1)
        x_attn = rearrange(x_attn, "b nh t h1 w1 d -> b t h1 w1 (nh d)")
        x = _x + + self.attn_out(x_attn)

        # ----------- MLP Layer -----------
        x = x.reshape(B, T * H * W, C) + self.mlp_out(mlp_h) # residual connection before attention part
        x = self._unrearrange_3d_natten(x, H, W)
        return x
    

class Neighborhood3DAttentionLayer_ResStyle(nn.Module):
    '''
    Local 3D Neighborhood Attention with ResNet Style Conv Out Layer
    '''
    def __init__(self, dim, heads, emb_dim, kernel_size, res, temporal_length=3, attn_dropout=0.0, proj_dropout = 0.0):
        super().__init__()
        self.n_heads = heads
        self.dim_head = dim // heads
        self.kernel_size = kernel_size

        self.scale = self.dim_head ** -0.5
        self.temporal_length = temporal_length
        
        # self.pos_emb = AxialRoPE(self.d_heads // 2, self.n_heads)
        self.attn_drop_rate = attn_dropout
        self.attn_dropout = nn.Dropout(self.attn_drop_rate)

        self.proj_drop_rate = proj_dropout

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.attn_out = zero_module(nn.Linear(dim, dim, bias=False))
        self.q_norm, self.k_norm = Normalize(self.dim_head), Normalize(self.dim_head)
        self.rope = RotaryEmbedding3D(self.dim_head, (self.temporal_length, res, res),)

        self.in_layers = nn.Sequential(
            NormalizeWithBias(dim),
            nn.SiLU(), # replace Conv with NATTEN
        )
        self.emb_layer = nn.Conv3d(emb_dim, dim * 2, kernel_size=(1, 1, 1))
        self.out_norm = NormalizeWithBias(dim)
        self.out_rest = nn.Sequential(
            nn.SiLU(),
            zero_module(
                nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            ),
        )
    
    def _rearrange_noise_emb_nattn(self, x: Tensor, emb: Tensor):
        b_t, c, h, w = x.shape # no need to reshape x
        B = b_t // self.temporal_length
        x = rearrange(x, '(b t) c h w -> b t c h w', b=B, h=h, w=w, t=self.temporal_length)
        emb = repeat(emb, "(b t) c -> b c t h w", b=B, h=h, w=w, t=self.temporal_length)
        return x, emb
    
    def forward(self, x, emb):
        x, emb = self._rearrange_noise_emb_nattn(x, emb)
        B, T, C, H, W = x.shape
        
        skip_term = x.permute(0, 1, 3, 4, 2)  # (B, T, H, W, C)
        _x = skip_term # (B, T, H, W, C), residual term before normalization

        # ----------- Initial Normalization -----------

        x_cthw = rearrange(skip_term, 'b t h w c -> b c t h w')
        x = self.in_layers(x_cthw) # GroupNorm
        
        # ----------- 3D Neighborhood Attention -----------
        x_flat = rearrange(x, 'b c t h w -> b (t h w) c')
        qkv = self.qkv_proj(x_flat)
        q, k, v = rearrange(qkv, "b (hw) (qkv h d) -> qkv b h hw d", qkv=3, h=self.n_heads)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)

        # Reshape for neighborhood attention: (b, t, h, w, heads, dim_head)
        q = rearrange(q, 'b nh (t h1 w1) d -> b nh t h1 w1 d', b=B, nh=self.n_heads, t=T, h1=H, w1=W)
        k = rearrange(k, 'b nh (t h1 w1) d -> b nh t h1 w1 d', b=B, nh=self.n_heads, t=T, h1=H, w1=W)
        v = rearrange(v, 'b nh (t h1 w1) d -> b nh t h1 w1 d', b=B, nh=self.n_heads, t=T, h1=H, w1=W)

        q = q * self.scale
        attn = natten.functional.na3d_qk(q, k, kernel_size=self.kernel_size, is_causal=False, dilation=1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        x_attn = natten.functional.na3d_av(attn, v, kernel_size=self.kernel_size, is_causal=False, dilation=1)
        x_attn = rearrange(x_attn, "b nh t h1 w1 d -> b t h1 w1 (nh d)")
        x = _x + + self.attn_out(x_attn)

        # ----------- Final Conv Layer -----------
        h = rearrange(x, 'b t h w c -> b c t h w')
        emb_out = self.emb_layer(emb)
        scale, shift = emb_out.chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_rest(h)

        h = rearrange(h, 'b c t h w -> (b t) c h w')
        _x = rearrange(_x, 'b t h w c -> (b t) c h w')
        return _x + h