import torch
from torch import nn, Tensor
import torch.nn.functional as F
from video_model.modules3d import zero_module
from einops import rearrange, repeat
from video_model.nattn3d import *
from video_model.u_vit3d import *


# inspired by https://github.com/jaketae/mlp-mixer/blob/master/mlp_mixer/core.py
class MLP(nn.Module):
    def __init__(self, num_features, expansion_factor, dropout):
        super().__init__()
        num_hidden = num_features * expansion_factor
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = zero_module(nn.Linear(num_hidden, num_features)) # zero init
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x    


class ChannelMixer(nn.Module):
    def __init__(self, channel_dim, expansion_factor, emb_dim, dropout):
        super().__init__()
        self.in_norm = nn.LayerNorm(channel_dim, elementwise_affine=False) # FiLM
        self.emb_layer = nn.Linear(emb_dim, channel_dim * 2)
        self.mlp = MLP(channel_dim, expansion_factor, dropout)

    def _rearrange_x_emb_channel_mixer(self, x: Tensor, emb: Tensor):
        x = rearrange(x , "b_t c h w -> b_t (h w) c")
        return x
    
    def _unrearrange_x_emb_channel_mixer(self, x: Tensor, H: int, W: int):
        x = rearrange(x , "b_t (h w) c -> b_t c h w", h = H, w = W)
        return x
    
    def forward(self, x: Tensor, emb: Tensor):
        BT, C, H, W = x.shape
        residual = x
        x = self._rearrange_x_emb_channel_mixer(x, emb)
        x = self.in_norm(x)
        emb_out = self.emb_layer(emb)
        scale, shift = emb_out.chunk(2, dim = -1)
        scale, shift = scale.unsqueeze(1), shift.unsqueeze(1)
        x = x * (1 + scale) + shift
        
        x_out = self.mlp(x)
        x_out = self._unrearrange_x_emb_channel_mixer(x_out, H, W)
        out = x_out + residual
        return out    


class UViT3D_Mixer(nn.Module):
    '''
    UViT3D-Mixer with Local Spatial/Temporal Attention + Channel Mixer
    At finer scales (x2): [ResBlock * 2, Mixer, Spatial Nattn, Temporal Attn, Mixer]
    At coarse scales (x2): [Global 3D Attn]
    '''

    def __init__(
            self,
            cfg: DictConfig,
            resolution: int,
            in_channels: int,
            out_channels: int,
            max_tokens: int,
            external_cond_dim: int,
            use_causal_mask=False,
    ):
        super().__init__()
        # ------------------------------- Configuration --------------------------------
        self.cfg = cfg
        self.external_cond_dim = external_cond_dim
        self.use_causal_mask = use_causal_mask
        
        # these configurations closely follow the notation in the SiD2 paper
        self.channels = cfg.channels
        self.emb_dim = cfg.emb_channels
        self.patch_size = cfg.patch_size
        self.block_types = cfg.block_types
        self.block_dropouts = cfg.block_dropouts
        self.num_updown_blocks = cfg.num_updown_blocks
        self.num_mid_blocks = cfg.num_mid_blocks
        num_heads = cfg.num_heads
        self.pos_emb_type = cfg.pos_emb_type
        self.num_levels = len(self.channels)
        self.is_transformers = [block_type != "ResBlock" for block_type in self.block_types]
        self.use_checkpointing = list(cfg.use_checkpointing)
        self.temporal_length = max_tokens

        # --------------- U-ViT module initialization ---------------
        # Input embedding and final projection.
        self.embed_input = EmbedInput(in_channels=in_channels, dim=self.channels[0], patch_size=self.patch_size,)
        self.project_output = ProjectOutput(dim=self.channels[0], out_channels=out_channels, patch_size=self.patch_size,)
        
        # ---------------- Noise-level embeddings ----------------------------
        self.noise_level_pos_embedding = ViTTimeEmbedding(
            dim=self.noise_level_dim,
            time_embed_dim=self.noise_level_emb_dim,
        )

        # --------------------------- Positional embeddings ----------------------------
        # We use a 1D learnable positional embedding or RoPE for every level with transformers (3D RoPE)
        assert self.pos_emb_type in ["learned_1d","rope",], f"Positional embedding type {self.pos_emb_type} not supported."
        self.pos_embs = nn.ModuleDict({})
        self.level_resolutions = {}

        for i_level, channel in enumerate(self.channels):
            level_resolution = resolution // self.patch_size // (2**i_level)
            self.level_resolutions[i_level] = level_resolution
            if not self.is_transformers[i_level]:
                continue
            pos_emb_cls, dim = None, None
            if self.pos_emb_type == "rope":
                pos_emb_cls = (RotaryEmbedding3D if self.block_types[i_level] == "TransformerBlock" else AxialRotaryEmbedding)
                dim = channel // num_heads
            else:
                pos_emb_cls = partial(SinusoidalPositionalEmbedding, learnable=True)
                dim = channel
            
            self.pos_embs[f"{i_level}"] = pos_emb_cls(dim, (self.temporal_length, level_resolution, level_resolution),)

        # ------------------------------- Blocks ------------------------------
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        block_type_to_cls = {
            "ResBlock": partial(ResBlock, emb_dim=self.emb_dim),
            "TransformerBlock": partial(
                TransformerBlock, emb_dim=self.emb_dim, heads=num_heads
            ),
        }
        
        # ------------------- Down-sampling Blocks -------------------
        for level_idx, (num_blocks, ch, block_type, block_dropout) in enumerate(
            zip(
                self.num_updown_blocks,
                self.channels[:-1],
                self.block_types[:-1],
                self.block_dropouts[:-1],
            )
        ):
            blocks = [
                block_type_to_cls[block_type](
                    ch, dropout=block_dropout, **self._get_rope_kwargs(level_idx)
                )
                for _ in range(num_blocks)
            ]

            # If this level uses ResBlock (i.e. not a transformer), append a local attention + mixer blocks.
            if block_type == "ResBlock":
                # Append one MIXER after the ResBlocks
                blocks.append(ChannelMixer(channel_dim=ch, expansion_factor=3, emb_dim=self.emb_dim, dropout=0.0))
                # Append local attention and temporal attention layers
                blocks.append(
                    NeighborhoodTransformerLayer(dim=ch, heads=num_heads, emb_dim=self.emb_dim,
                                                 kernel_size=7, res=self.level_resolutions[level_idx])
                )
                blocks.append(
                    TemporalAttentionBlock(dim=ch, heads=num_heads, emb_dim=self.emb_dim, 
                                           temporal_length = self.temporal_length)
                )
                # Append one MIXER after local attention
                blocks.append(ChannelMixer(channel_dim=ch, expansion_factor=3, emb_dim=self.emb_dim, dropout=0.0))
                
            # Append downsampling operation.
            blocks.append(Downsample(ch, self.channels[level_idx + 1]))
            self.down_blocks.append(nn.ModuleList(blocks))
        
        # --------------------- Middle Blocks ---------------------
        self.mid_blocks = nn.ModuleList(
            [block_type_to_cls[self.block_types[-1]](
                    self.channels[-1],
                    dropout=self.block_dropouts[-1],
                    **self._get_rope_kwargs(self.num_levels - 1),)
                for _ in range(self.num_mid_blocks)
            ]
        )        

        # --------------------- Up-sampling Blocks ---------------------
        self.up_blocks = nn.ModuleList()
        for rev_idx, (num_blocks, ch, block_type, block_dropout) in enumerate(
            zip(
                reversed(self.num_updown_blocks),
                reversed(self.channels[:-1]),
                reversed(self.block_types[:-1]),
                reversed(self.block_dropouts[:-1]),
            )
        ):
            level_idx = self.num_levels - 2 - rev_idx
            blocks = (
                [Upsample(self.channels[level_idx + 1], ch)]
                + [block_type_to_cls[block_type](
                        ch, dropout=block_dropout, **self._get_rope_kwargs(level_idx))
                    for _ in range(num_blocks)
                ]
            )

            if block_type == "ResBlock":
                # Append one MIXER after the ResBlocks
                blocks.append(ChannelMixer(channel_dim=ch, expansion_factor=3, emb_dim=self.emb_dim, dropout=0.0))
                # Append local attention and temporal attention layers
                blocks.append(
                    NeighborhoodTransformerLayer(dim=ch, heads=num_heads, emb_dim=self.emb_dim,
                                                 kernel_size=7, res=self.level_resolutions[level_idx])
                )
                blocks.append(
                    TemporalAttentionBlock(dim=ch, heads=num_heads, emb_dim=self.emb_dim, 
                                           temporal_length = self.temporal_length)
                )
                # Append one MIXER after the ResBlocks
                blocks.append(ChannelMixer(channel_dim=ch, expansion_factor=3, emb_dim=self.emb_dim, dropout=0.0))
            
            self.up_blocks.append(nn.ModuleList(blocks))  

    @property
    def noise_level_dim(self) -> int:
        return 256

    @property
    def noise_level_emb_dim(self) -> int:
        return self.emb_dim

    def _get_rope_kwargs(self, level_idx: int) -> Dict[str, Any]:
        """
        Returns keyword arguments for rotary positional embeddings if applicable.
        """
        if self.pos_emb_type == "rope" and self.is_transformers[level_idx]:
            return {"rope": self.pos_embs[str(level_idx)]}
        return {}      

    def _rearrange_and_add_pos_emb_if_transformer(self, x: Tensor, emb: Tensor, level_idx: int) -> Tuple[Tensor, Tensor]:
        """
        Rearranges tensor for transformer blocks and adds positional embeddings if needed.
        """
        if not self.is_transformers[level_idx]:
            return x, emb

        b_t, _, h, w = x.shape  # x shape: (B*T, C, H, W)
        x = rearrange(x, "(b t) c h w -> b (t h w) c", t=self.temporal_length)
        if self.pos_emb_type == "learned_1d":
            x = self.pos_embs[str(level_idx)](x)
        emb = repeat(emb, "(b t) c -> b (t h w) c", t=self.temporal_length, h=h, w=w)
        return x, emb

    def _unrearrange_if_transformer(self, x: Tensor, level_idx: int) -> Tensor:
        """
        Rearranges tensor back to its original shape if transformer-specific rearrangement was applied.
        """
        if not self.is_transformers[level_idx]:
            return x
        h_w = int((x.shape[1] / self.temporal_length) ** 0.5)
        x = rearrange(x, "b (t h w) c -> (b t) c h w", t=self.temporal_length, h=h_w, w=h_w)
        return x
    
    def _run_level_blocks(self, x: Tensor, emb: Tensor, i_level: int, is_up: bool = False) -> Tensor:
        """
        Run the blocks (except up/downsampling blocks) for a given level.
        """
        blocks = (
            self.mid_blocks
            if i_level == self.num_levels - 1
            else (
                self.up_blocks[self.num_levels - 2 - i_level][1:]
                if is_up
                else self.down_blocks[i_level][:-1]
            )
        )
        for block in blocks:
            x = block(x, emb)
        return x
    
    def _run_level(self, x: Tensor, emb: Tensor, i_level: int, is_up: bool = False) -> Tensor:
        """
        Run the blocks (except up/downsampling blocks) for a given level, accompanied by reshaping operations before and after.
        """
        x, emb = self._rearrange_and_add_pos_emb_if_transformer(x, emb, i_level)
        x = self._run_level_blocks(x, emb, i_level, is_up)
        x = self._unrearrange_if_transformer(x, i_level)
        return x

    def forward(
        self,
        x: Tensor,
        noise_levels: Tensor,
        x_cond: Tensor = None,
    ) -> Tensor:
        """
        Forward pass of the U-ViT3D-Mixer backbone.
        Args:
            x: Input tensor of shape (B, T, C, H, W).
            noise_levels: Noise level tensor of shape (B, T).
            external_cond: External conditioning tensor of shape (B, T, C).
        Returns:
            Output tensor of shape (B, T, C, H, W).
        """

        if x_cond is not None:
            x = torch.cat((x_cond, x), dim = 2) # batch, frame, channel, h, w

        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.embed_input(x)

        # Embeddings
        emb = self.noise_level_pos_embedding(noise_levels)
        emb = rearrange(emb, "b t c -> (b t) c")

        hs_before = []  # hidden states before downsampling
        hs_after = []  # hidden states after downsampling

        # Down-sampling blocks
        for i_level, down_block in enumerate(self.down_blocks,):
            x = self._run_level(x, emb, i_level)
            hs_before.append(x)
            x = down_block[-1](x)
            hs_after.append(x)

        # Middle blocks
        x = self._run_level(x, emb, self.num_levels - 1)

        # Up-sampling blocks
        for _i_level, up_block in enumerate(self.up_blocks):
            i_level = self.num_levels - 2 - _i_level
            x = x - hs_after.pop()
            x = up_block[0](x) + hs_before.pop()
            x = self._run_level(x, emb, i_level, is_up=True)

        x = self.project_output(x)
        return rearrange(x, "(b t) c h w -> b t c h w", t=self.temporal_length)
    