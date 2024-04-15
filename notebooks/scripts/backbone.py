#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Callable
from einops import rearrange, repeat
from torch import nn, Tensor
from torchvision import transforms, models

#torch._dynamo.config.verbose = True
torch.cuda.empty_cache()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#------------------------------------------------
#   CNN
#------------------------------------------------

class SelfAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 1, head_channels: int = None):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        hidden_dim = in_channels * num_heads if head_channels is None else head_channels * num_heads
        # query-key-value
        self.qkv = nn.Conv2d(in_channels, hidden_dim * 3, 1)
        self.output = nn.Sequential(nn.Conv2d(hidden_dim, out_channels, 1), nn.GroupNorm(1, out_channels))
        self.activation = nn.ReLU()

    def forward(self, x):
        h, w = x.shape[-2:]
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.num_heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn, bhen -> bhde', k, v)
        context = torch.einsum('bhde, bhdn -> bhen', context, q)
        context = rearrange(context, 'b heads c (h w) -> b (heads c) h w', heads=self.num_heads, w=w, h=h)
        return torch.tanh(self.activation(self.output(context)))

        
class BridgeAttention(nn.Module):
    def __init__(self, channels: int):
        super(BridgeAttention, self).__init__()
        self.wx = nn.Conv2d(channels, channels, 1, padding=0)
        self.wg = nn.Conv2d(channels, channels, 1, padding=0)
        self.activation = nn.ReLU()
        self.attn = nn.Conv2d(channels, channels, 1, padding=0)
        
    def forward(self, pass_through, gaiting_signal):
        x = self.wx(pass_through)
        g = self.wg(gaiting_signal)
        x = torch.tanh(self.attn(self.activation(x + g)))
        return pass_through * (x + 1)


class ConvNorm(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, padding: int = 1):
        super(ConvNorm, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding),
            nn.GroupNorm(1, out_channels))

        
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, residual: bool = True, attn: bool = True):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            ConvNorm(in_channels, out_channels),
            nn.GELU(),
            ConvNorm(out_channels, out_channels))
        self.residual = nn.Conv2d(in_channels, out_channels, 1, padding=0) if residual else None
        self.attn = SelfAttention(in_channels, out_channels) if attn is not None else None
        self.activation = nn.ReLU()

    def forward(self, x):
        attn = self.attn(x) if self.attn is not None else None
        output = self.block(x)
        if self.residual:
            output = output + self.residual(x)
        # either attention or activation
        return self.activation(output) if attn is None else output * attn


class DownsampleBlock(nn.Module):        
    def __init__(self, in_channels: int, out_channels: int, residual: bool = True, attn: bool = True):
        super(DownsampleBlock, self).__init__()
        self.block = ConvBlock(in_channels, out_channels, residual, attn)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        pass_through = self.block(x)
        output = self.pool(pass_through)
        return output, pass_through


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, channels: int = 64, depth: int = 4,
                       residual: bool = True, attn: bool = True):
        super(CNNEncoder, self).__init__()
        self.channels = channels
        self.depth = depth
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(DownsampleBlock(in_channels, channels, residual, attn))
            in_channels, channels = channels, channels * 2
        self.residual = residual
        self.attn = attn
        
    def forward(self, x):
        outputs = []
        for block in self.blocks:
            x, pass_through = block(x)
            outputs.append(pass_through)
        return outputs


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, residual: bool, attn: bool = True,
                 bridge: bool = True, bridge_attn: bool = True):
        super(UpsampleBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.bridge_attn = BridgeAttention(out_channels) if bridge and bridge_attn else None
        self.block = ConvBlock(in_channels if bridge else in_channels//2, out_channels, residual, attn)
        self.bridge = bridge

    def forward(self, x, pass_through=None):
        x = self.deconv(x)
        if self.bridge: # use skip-connection
            if self.bridge_attn: # apply attention-gate to skip-connection
                pass_through = self.bridge_attn(pass_through, x)
            x = torch.cat((pass_through, x), dim=1)
        return self.block(x)
    

class CNNDecoder(nn.Module):
    def __init__(self, in_channels: int, output_dim: int, depth: int, residual: bool, attn: bool,
                       bridge: bool, bridge_attn: bool):
        super(CNNDecoder, self).__init__()
        self.blocks = nn.ModuleList()
        out_channels = in_channels//2
        for _ in range(depth):
            self.blocks.append(UpsampleBlock(in_channels, out_channels, residual, attn, bridge, bridge_attn))
            in_channels, out_channels = out_channels, out_channels//2
        self.head = nn.Conv2d(in_channels, output_dim, 1, padding=0) # 1x1 convolution
        
    def forward(self, outputs):
        assert len(outputs) == len(self.blocks) + 1
        outputs = list(outputs)
        x = outputs.pop()
        for block in self.blocks:
            x = block(x, outputs.pop())
        return self.head(x)
    
    
class UpsampleBlockNoBridge(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, residual: bool = True, attn: bool = True):
        super().__init__(
            nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=4, stride=2),
            ConvBlock(in_channels//2, out_channels, residual, attn))
    

class CNNDecoderNoBridge(nn.Module):
    def __init__(self, in_channels: int, output_dim: int, depth: int, residual: bool, attn: bool):
        super().__init__()
        self.blocks = nn.ModuleList()
        out_channels = in_channels//2
        for _ in range(depth):
            self.blocks.append(UpsampleBlockNoBridge(in_channels, out_channels, residual, attn))
            in_channels, out_channels = out_channels, out_channels//2
        self.head = nn.Conv2d(in_channels, output_dim, 1, padding=0) # 1x1 convolution
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)

#------------------------------------------------
#   ViT
#------------------------------------------------
    
class ViewToSequence(nn.Module):
    def __init__(self,
                 view_size: int,
                 patch_size: int,
                 embed_size: int,
                 semantic_dim: int = 0,
                 channels: int = 1):
        
        super(ViewToSequence, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(channels, embed_size, kernel_size=patch_size, stride=patch_size)
        # conditional and other tokens
        self.tokens = nn.Parameter(torch.randn(1, semantic_dim, embed_size)) if semantic_dim > 0 else None
        self.positions = nn.Parameter(torch.randn((view_size // patch_size) ** 2 + semantic_dim, embed_size))
                
    def forward(self, x):
        b = x.shape[0]
        # patch-sequence: either linear or conv
        x = self.projection(x)
        x = rearrange(x, 'b e (h) (w) -> b (h w) e')
        if not self.tokens is None:
            tokens = repeat(self.tokens, '() n e -> b n e', b=b)
            # prepend the tokens to the input
            x = torch.cat([tokens, x], dim=1)
        # add positional embedding
        x += self.positions
        return x
    
    
class SequenceToView(nn.Module):
    def __init__(self,
                 view_size: int,
                 patch_size: int,
                 embed_size: int,
                 semantic_dim: int = 0,
                 channels: int = 1):
        
        super(SequenceToView, self).__init__()
        self.patch_size = patch_size
        self.semantic_dim = semantic_dim
        self.projection = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, patch_size ** 2 * channels, bias=True))
        # preventing artifacts
        #self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x):
        x = self.projection(x)
        x = x[:, self.semantic_dim:, :] # skip tokens
        d, p = int(x.shape[1] ** 0.5), self.patch_size
        return rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=d, w=d, p1=p, p2=p)
        #return self.conv(x)
    
    
class Attention(nn.Module):
    def __init__(self,
                 embed_size: int,
                 num_heads: int = 4):
        
        super(Attention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        # queries, keys, values in one matrix
        self.qkv = nn.Linear(embed_size, embed_size * 3, bias=False)
        self.projection = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU()) # selection (remove distractions)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        q, k, v = rearrange(self.qkv(x), 'b n (h d qkv) -> (qkv) b h n d', h=self.num_heads, qkv=3)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', q, k) # batch, num_heads, query_len, key_len
        if mask is not None:
            energy.mask_fill(~mask, torch.finfo(torch.float32).min)
            
        scaling = self.embed_size ** 0.5
        att = torch.softmax(energy, dim=-1) / scaling
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.projection(out)
    
    
class MLP(nn.Sequential):
    def __init__(self, embed_size: int, expansion: int = 4):
        super(MLP, self).__init__(
            nn.Linear(embed_size, expansion * embed_size),
            nn.GELU(),
            nn.Linear(expansion * embed_size, embed_size))
    
    
class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_size: int,
                 bridge: bool,
                 expansion: int = 4):
        
        super(TransformerBlock, self).__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(embed_size),
            Attention(embed_size))
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_size),
            MLP(embed_size, expansion=expansion))
        
        self.merge = nn.Linear(2 * embed_size, embed_size) if bridge else None
            
    def forward(self, x, pass_through=None):
        if self.merge is not None:
            x = self.merge(torch.cat((pass_through, x), dim=2))
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x
    
    
class TransformerEncoder(nn.Module):
    def __init__(self,
                 view_size: int,
                 patch_size: int,
                 embed_size: int,
                 depth: int,
                 expansion: int = 4):
        
        super(TransformerEncoder, self).__init__()
        # patch embed
        self.sequence = ViewToSequence(view_size, patch_size, embed_size)
        # down-blocks
        self.blocks = nn.ModuleList([TransformerBlock(embed_size, False, expansion) for _ in range(depth)])
        self.depth = depth
        self.view_size = view_size
        self.patch_size = patch_size
        self.embed_size = embed_size
                
    def forward(self, x):
        x = self.sequence(x)
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return outputs
    
    
class TransformerDecoder(nn.Module):
    def __init__(self,
                 view_size: int,
                 patch_size: int,
                 embed_size: int,
                 depth: int,
                 channels: int = 1,
                 bridge: bool = True,
                 expansion: int = 4):
        
        super(TransformerDecoder, self).__init__()
        # up-blocks
        self.blocks = nn.ModuleList([TransformerBlock(embed_size, bridge, expansion) for _ in range(depth)])
        self.unpatch = SequenceToView(view_size, patch_size, embed_size, channels=channels)
        
    def forward(self, outputs):
        assert len(outputs) == len(self.blocks) + 1
        x = outputs.pop()
        for block in self.blocks:
            x = block(x, outputs.pop())
        return self.unpatch(x)

#------------------------------------------------
#   GPT
#------------------------------------------------
    
class AttentionHead(nn.Module):
    def __init__(self, embed_size, seq_length, head_size, dropout: float = 0.5):
        super().__init__()
        self.seq_length = seq_length
        self.embed_size = embed_size
        self.head_size = head_size
        
        self.key   = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.query = nn.Linear(self.embed_size, self.head_size, bias=False)        
        self.value = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer('tril', torch.tril(torch.ones(self.seq_length, self.seq_length)))        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, t, c = x.shape
        K, Q, V = self.key(x), self.query(x), self.value(x)
        W = Q @ K.transpose(-2, -1) * (c ** 0.5)
        W = W.masked_fill(self.tril[:t,:t]==0, float('-inf'))
        W = self.softmax(W)
        W = self.dropout(W)
        return W @ V

    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, seq_length, num_heads, dropout: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        head_size = embed_size // num_heads        
        self.heads = nn.ModuleList([AttentionHead(embed_size, seq_length, head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(embed_size, embed_size)        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.projection(x)
        x = self.dropout(x)
        return x
    
    
class GPTBlock(nn.Module):
    def __init__(self, embed_size, seq_length, num_heads, expansion: int = 4, dropout: float = 0.5):
        super().__init__()
        self.attn = MultiHeadAttention(embed_size, seq_length, num_heads)
        self.attn_norm = nn.LayerNorm(embed_size)
        self.feed = nn.Sequential(
            nn.Linear(embed_size, embed_size * expansion),
            nn.ReLU(),
            nn.Linear(embed_size * expansion, embed_size),
            nn.Dropout(dropout))
        self.feed_norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.feed(self.feed_norm(x))        
        return x
    
    
class ContentGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, seq_length, num_heads, num_blocks: int = 3):
        super().__init__()        
        self.embed_size = embed_size
        self.seq_length = seq_length        
        self.token_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed   = nn.Embedding(seq_length, embed_size)        
        self.blocks = nn.Sequential(*[GPTBlock(embed_size, seq_length, num_heads)] * num_blocks,
                                    nn.LayerNorm(embed_size) )
        self.lm_head = nn.Linear(embed_size, vocab_size)
        
    def forward(self, x):
        E, P = self.token_embed(x), self.pos_embed(torch.arange(x.shape[1], device=DEVICE))
        return self.lm_head(self.blocks(E + P))
    
    def generate(self, x, total):
        for _ in range(total):
            context = x[:,-self.seq_length:]
            logits= self(context)
            logits = logits[:,-1,:]
            proba = torch.softmax(logits, dim=-1)
            x = torch.cat((x, torch.multinomial(proba, num_samples=1)), dim=1)
        return x    
    
#------------------------------------------------
#   utilities
#------------------------------------------------
    
class MeanReduce(nn.Module):
    def forward(self, x):
        return torch.mean(x, axis=1)
    
    
class VisualEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, reduce: nn.Module = None, frozen: bool = True):
        super(VisualEncoder, self).__init__()
        self.encoder = backbone
        if frozen: # freeze weights
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.reduce = reduce
        
    def forward(self, x):
        # our unet-encoder returns list of outputs from all the levels --
        # here we only need the bottleneck
        x = self.encoder(x).pop()
        if self.reduce is None:
            return torch.flatten(x, start_dim=1)
        return torch.flatten(self.reduce(x), start_dim=1)
    
    
class Projection(nn.Module):
    def __init__(self, embedding_dim: int, latent_dim: int):
        super(Projection, self).__init__()
        self.projection = nn.Linear(embedding_dim, latent_dim)
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim))
        self.norm = nn.LayerNorm(latent_dim)
    
    def forward(self, x):
        h = self.projection(x)
        x = self.mlp(h)
        x = x + h
        return self.norm(x)
    
    
class VisualProjection(nn.Sequential):
    def __init__(self, encoder: nn.Module, embedding_dim: int, latent_dim: int, dropout: float = 0.):
        super(VisualProjection, self).__init__(
            encoder,
            Projection(embedding_dim, latent_dim))

        
class Head(nn.Sequential):
    def __init__(self, latent_dim: int, output_dim: int, activation: nn.Module = nn.Identity()):
        super(Head, self).__init__(
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim),
            activation)

        
class MultitaskClassifier(nn.Module):
    def __init__(self, latent_dim: int, tasks: list):
        super(MultitaskClassifier, self).__init__()
        self.tasks = nn.ModuleList([Head(latent_dim, num_clases) for num_clases in tasks])
                                         
    def forward(self, x):
        return [task(x) for task in self.tasks]

