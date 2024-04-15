#!/usr/bin/env python
# coding: utf-8

#----------------------------------------------------------------
# VAE-zoo training with `contrast` term to prevent mode-collapse
#               https://github.com/arcta/ae-lego
#----------------------------------------------------------------

import torch
import numpy as np
import pandas as pd

from typing import Optional
from torch import nn, Tensor
from torch.distributions import OneHotCategorical, RelaxedOneHotCategorical
#from torchsummary import summary

from .backbone import *


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DIM = 512


def get_encoder(depth=4):
    return CNNEncoder(channels=64, depth=depth, residual=True, attn=True).to(DEVICE)


def get_decoder(depth=3):
    return CNNDecoderNoBridge(DIM, 1, depth, residual=True, attn=True).to(DEVICE)


class Neck(nn.Module):
    """
    Takes our original encoder input and optional semantic context vector
    and converts them to a vector of a target latent space size
    
    We intentianally create a separate projection for the context injection:
    direct access to that leyer cold be useful for some other modules in the system or diagnostics
    """
    def __init__(self, hidden_dim: int, semantic_dim: int = 0):
        super().__init__()
        self.semantic_dim = semantic_dim
        # adapt our specific encoder output
        self.adapter = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim = 1))
        # inject context if present
        self.context_projection = nn.Sequential(nn.Linear(semantic_dim + DIM, DIM), nn.ReLU())
        # compress to target latent space dim
        self.latent_projection = nn.Linear(DIM, hidden_dim)
        
    def forward(self, x, context: Optional[Tensor] = None):
        h = self.adapter(x[-1])
        if self.semantic_dim > 0 and not context is None:
            h = self.context_projection(torch.cat((h, context), 1))
        return self.latent_projection(h)


class AutoEncoder(nn.Module):
    """
    Takes our existing encoder and decoder and connects them
    via Neck module disabled semantic channel: all inputs except of visual are simply ignored
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.neck = Neck(DIM)
        
    def encode(self, x):
        return self.neck(self.encoder(x))
    
    def forward(self, x, **kwargs):
        h = self.encode(x)
        return self.decoder(h.view(*h.size(), 1, 1))
    
    
class ZEncoder(nn.Module):
    """
    Enhance our existing encoder with a sense of proper data distribution
    and allow some optional arbitrary semantic channel interact with it
    """
    def __init__(self, encoder: nn.Module, latent_dim: int, semantic_dim: int = 0):
        super().__init__()
        self.latent_dim = latent_dim
        self.semantic_dim = semantic_dim
        self.encoder = encoder
        self.neck = Neck(DIM, semantic_dim)
        # reparametrization layers
        self.mean = nn.Linear(DIM, latent_dim)
        self.logvar = nn.Linear(DIM, latent_dim)
        
    def encode(self, x, context: Optional[Tensor] = None):
        return self.neck(self.encoder(x), context)

    def projection(self, x):
        return self.mean(x), self.logvar(x)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x, context: Optional[Tensor] = None):
        h = self.encode(x, context)
        mean, logvar = self.projection(h)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar

    
class Condition(nn.Module):
    """
    Takes latent representation input and optional semantic context vector
    and converts them to a decoder input
    
    Conceptually the same as Neck: a separate projection for the context injection to provide
    direct access to that leyer will be useful for state diagnostics during the training
    """
    def __init__(self, input_dim: int, output_dim: int, semantic_dim: int = 0):
        super().__init__()
        self.semantic_dim = semantic_dim
        # inject context if any
        self.context_projection = nn.Linear(semantic_dim + input_dim, input_dim)
        # adjust to the required output format
        self.output_projection = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU())
        
    def forward(self, z, context: Optional[Tensor] = None):
        if self.semantic_dim > 0 and not context is None:
            z = self.context_projection(torch.cat((z, context), 1))
        # return conditioned latent representation along with adjusted output for diagnostics
        return self.output_projection(z), z
    
    
class ZDecoder(nn.Module):
    """
    Enhance our existing decoder with ability to translate
    conditional latent representation into its native input
    """
    def __init__(self, decoder: nn.Module, latent_dim: int, semantic_dim: int = 0):
        super().__init__()
        self.latent_dim = latent_dim
        self.semantic_dim = semantic_dim
        self.decoder = decoder
        # make a conditional adapter to our encoder and decoder heads
        self.condition = Condition(latent_dim, DIM, semantic_dim)
    
    def forward(self, z, context: Optional[Tensor] = None):
        h, c = self.condition(z, context)
        h = h.view(h.size(0), DIM, 1, 1)
        return self.decoder(h), c
    
    
class VAE(nn.Module):
    """
    Take our existing encoder and decoder and build VAE connector
    with optional noisy semantic channel
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int,
                 encoder_semantic_dim: int = 0, decoder_semantic_dim: int = 0, tau: float = 0.,
                 mlp: nn.Module = None):
        super().__init__()
        self.latent_dim = latent_dim
        # adjust encoder and decoder
        self.encoder = ZEncoder(encoder, latent_dim, encoder_semantic_dim)
        self.decoder = ZDecoder(decoder, latent_dim, decoder_semantic_dim)
        self.mlp = mlp
        # make noise balansing parameter for semantic channels
        self.tau = nn.Parameter(torch.tensor([tau]))
        # tag output for local convenience
        self.keys = ['rec','z','mean','log-var','tau','z-context','label']
        
    @staticmethod
    def translate(semantic_dim, context, temperature):
        """
        for different scenarios:
        unsupervised arbitrary floating point semantic vector goes as is
        while labels need encoding, and maybe noise if semantic channel set to be noisy
        """
        if semantic_dim == 0:
            return None
        if context is None or torch.is_floating_point(context):
            return context
        context = F.one_hot(context, num_classes=semantic_dim).float().to(DEVICE)
        if temperature is not None:
            context += torch.rand(*context.size()).to(DEVICE) * temperature * 0.25
        return context
            
    def forward(self, x, context: Optional[Tensor] = None, temperature: Optional[float] = None):
        if temperature is None:
            temperature = torch.exp(self.tau)
        # representation and dist-params
        z, mean, logvar = self.encoder(x, self.translate(self.encoder.semantic_dim, context, temperature))
        # reconstruction and its source conditioned representation
        rec, zcontext = self.decoder(z, self.translate(self.decoder.semantic_dim, context, temperature))
        label = None if self.mlp is None else self.mlp(z)
        return rec, z, mean, logvar, self.tau, zcontext, label
    
    @torch.no_grad()
    def sample(self, num_samples, context: Optional[Tensor] = None, temperature: Optional[float] = 1.):
        if self.decoder.semantic_dim == 0:
            context = None
        if type(context) == int:
            # convert single int data label to labels num_samples batch
            context = (torch.ones(num_samples) * context).long().to(DEVICE)
        # translate context as model semantics state (temperature None)
        context = self.translate(self.decoder.semantic_dim, context, None)
        # generate noise with temperature
        z = torch.randn(num_samples, self.latent_dim).to(DEVICE) * temperature
        return self.decoder(z, context)[0]
    
    
class GEncoder(nn.Module):
    """
    Enhance our existing encoder with a sense of proper data distribution including categorical
    and allow some optional semantic channel interact with it
    """
    def __init__(self, encoder: nn.Module, latent_dim: int, categorical_dim: int, semantic_dim: int = 0):
        super().__init__()
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.semantic_dim = semantic_dim
        self.encoder = encoder
        self.neck = Neck(latent_dim * categorical_dim, semantic_dim)
        
    def encode(self, x, context: Optional[Tensor] = None):
        return self.neck(self.encoder(x), context)

    def gumbel_sample(self, shape, eps=1e-10):
        """
        scale 0 location 1 Gumbel dist: -log(-log(U(0,1))) 
        """
        return -torch.log(-torch.log(torch.rand(shape).to(DEVICE) + eps) + eps)

    def gumbel_softmax(self, x, temperature):
        """
        add Gumbel noise to the logits (x), apply temperature and softmax
        """
        h = x + self.gumbel_sample(x.size())
        h = torch.softmax(h / temperature, dim=-1)
        return h.view(-1, self.latent_dim * self.categorical_dim)
    
    def forward(self, x, context: Optional[Tensor] = None, temperature: Optional[float] = 1.):
        h = self.encode(x, context)
        h = h.view(h.size(0), self.latent_dim, self.categorical_dim)
        # posterior and prior
        q = torch.softmax(h, dim=-1).view(h.size(0), self.latent_dim * self.categorical_dim)
        p = self.gumbel_softmax(h, temperature)
        return p, q
    
    
class DVAE(nn.Module):
    """
    Take our existing encoder and decoder and build DVAE connector
    with optional noisy semantic channel
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int, categorical_dim: int,
                       encoder_semantic_dim: int = 0, decoder_semantic_dim: int = 0, tau: float = 1.,
                       mlp: nn.Module = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        # noise balansing parameter
        self.tau = nn.Parameter(torch.tensor([tau]))
        # adjusted encoder and decoder
        self.encoder = GEncoder(encoder, latent_dim, categorical_dim, encoder_semantic_dim)
        self.decoder = ZDecoder(decoder, latent_dim * categorical_dim, decoder_semantic_dim)
        self.mlp = mlp
        # output tags
        self.keys = ['rec','q','p','tau','p-context','label']
        
    @staticmethod
    def translate(semantic_dim, context, temperature):
        """
        for different scenarios:
        unsupervised arbitrary floating point semantic vector goes as is
        while labels need encoding, and maybe noise if semantic channel set to be noisy
        """
        if semantic_dim == 0:
            return None
        if context is None or torch.is_floating_point(context):
            return context
        context = F.one_hot(context, num_classes=semantic_dim).to(DEVICE)
        if temperature is None:
            return context
        return RelaxedOneHotCategorical(temperature, probs=context).sample()
            
    def forward(self, x, context: Optional[Tensor] = None, temperature: Optional[float] = None):
        if temperature is None:
            temperature = torch.exp(self.tau)
        # representation: prior and posterior
        p, q = self.encoder(x, self.translate(self.encoder.semantic_dim, context, temperature), temperature)
        # reconstruction and its source conditioned representation
        rec, pcontext = self.decoder(p, self.translate(self.decoder.semantic_dim, context, temperature))
        label = None if self.mlp is None else self.mlp(p)
        return rec, p, q, self.tau, pcontext, label
    
    @torch.no_grad()
    def sample(self, num_samples, context: Optional[Tensor] = None, temperature: Optional[float] = 1.):
        if self.decoder.semantic_dim == 0:
            context = None
        if type(context) == int:
            context = (torch.ones(num_samples) * context).long().to(DEVICE)
        # condition without semantic noise
        context = self.translate(self.decoder.semantic_dim, context, None)
        d = self.latent_dim * self.categorical_dim
        probs = torch.ones((num_samples, self.latent_dim, self.categorical_dim))/self.categorical_dim
        if temperature is None:
            p = OneHotCategorical(probs=probs.to(DEVICE)).sample().view(num_samples, d)
        else:
            p = RelaxedOneHotCategorical(temperature, probs=probs.to(DEVICE)).sample().view(num_samples, d)
        return self.decoder(p, context)[0]

    