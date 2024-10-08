from functools import wraps

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

from torch_cluster import fps

from timm.models.layers import DropPath

from meshgen.utils.ops import bacthed_fps


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, drop_path_rate=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Linear(dim * mult, dim)
        )

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x):
        return self.drop_path(self.net(x))


class Attention(nn.Module):
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, drop_path_rate=0.0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x, context=None, mask=None):
        # h = self.heads

        # q = self.to_q(x)
        # context = default(context, x)
        # k, v = self.to_kv(context).chunk(2, dim=-1)

        # q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        # sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        # if exists(mask):
        #     mask = rearrange(mask, "b ... -> b (...)")
        #     max_neg_value = -torch.finfo(sim.dtype).max
        #     mask = repeat(mask, "b j -> (b h) () j", h=h)
        #     sim.masked_fill_(~mask, max_neg_value)

        # # attention, what we cannot get enough of
        # attn = sim.softmax(dim=-1)

        # out = einsum("b i j, b j d -> b i d", attn, v)
        # out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        # return self.drop_path(self.to_out(out))

        return self.flash_forward(x, context, mask)

    def flash_forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        # sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        # if exists(mask):
        #     mask = rearrange(mask, "b ... -> b (...)")
        #     max_neg_value = -torch.finfo(sim.dtype).max
        #     mask = repeat(mask, "b j -> (b h) () j", h=h)
        #     sim.masked_fill_(~mask, max_neg_value)

        # # attention, what we cannot get enough of
        # attn = sim.softmax(dim=-1)

        # out = einsum("b i j, b j d -> b i d", attn, v)
        # out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.drop_path(self.to_out(out))


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack(
            [
                torch.cat(
                    [
                        e,
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        e,
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                        e,
                    ]
                ),
            ]
        )
        self.register_buffer("basis", e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim + 3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum("bnd,de->bne", input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(
            torch.cat([self.embed(input, self.basis), input], dim=2)
        )  # B x N x C
        return embed


class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.mean.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.mean.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            reduce_dims = list(range(1, len(self.mean.shape)))
            if other is None:
                return 0.5 * torch.mean(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=reduce_dims,
                )
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=reduce_dims,
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


class AutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=2048,
        num_latents=512,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
        learnable_query=False,
        ordering=False,
        ln_post=False,
    ):
        super().__init__()

        self.depth = depth

        self.ndim = dim
        self.num_inputs = num_inputs
        self.num_latents = num_latents
        self.learnable_query = learnable_query
        self.ordering = ordering

        if self.learnable_query:
            self.query = nn.Parameter(torch.randn(1, num_latents, queries_dim))

        self.cross_attend_blocks = nn.ModuleList(
            [
                PreNorm(
                    dim, Attention(dim, dim, heads=1, dim_head=dim), context_dim=dim
                ),
                PreNorm(dim, FeedForward(dim)),
            ]
        )

        self.point_embed = PointEmbed(dim=dim)

        get_latent_attn = lambda: PreNorm(
            dim, Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1)
        )
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}

        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [get_latent_attn(**cache_args), get_latent_ff(**cache_args)]
                )
            )

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            Attention(queries_dim, dim, heads=1, dim_head=dim),
            context_dim=dim,
        )
        self.decoder_ff = (
            PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        )

        self.to_outputs = (
            nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()
        )

        if ln_post:
            self.ln_post = nn.LayerNorm(dim)
            self.output_ln_post = nn.LayerNorm(dim)
        else:
            self.ln_post = nn.Identity()
            self.output_ln_post = nn.Identity()

    def encode(self, pc):
        # pc: B x N x 3
        B, N, D = pc.shape
        assert N == self.num_inputs

        if not self.learnable_query:
            ###### fps

            flattened = pc.view(B * N, D)

            batch = torch.arange(B).to(pc.device)
            batch = torch.repeat_interleave(batch, N)

            pos = flattened

            ratio = 1.0 * self.num_latents / self.num_inputs

            idx = fps(pos, batch, ratio=ratio)

            sampled_pc = pos[idx]
            sampled_pc = sampled_pc.view(B, -1, 3)

            # idx = bacthed_fps(pc, self.num_latents)

            # if self.ordering:
            #     idx = torch.sort(idx, dim=1)[0]
            # ######
            # # sampled_pc = pc[idx]
            # sampled_pc = torch.gather(pc, 1, idx.unsqueeze(-1).expand(-1, -1, 3))

            sampled_pc_embeddings = self.point_embed(sampled_pc)
        else:
            sampled_pc_embeddings = self.query

        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = (
            cross_attn(sampled_pc_embeddings, context=pc_embeddings, mask=None)
            + sampled_pc_embeddings
        )
        x = cross_ff(x) + x

        x = self.ln_post(x)

        return x

    def decode(self, x, queries):

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # cross attend from decoder queries to latents
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context=x)

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        latents = self.output_ln_post(latents)

        return self.to_outputs(latents)

    def forward(self, pc, queries):
        x = self.encode(pc)

        o = self.decode(x, queries).squeeze(-1)

        return {"logits": o}


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        depth=8,
        dim=512,
        queries_dim=3072,
        output_dim=1,
        num_inputs=2048,
        num_latents=512,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
        learnable_query=True,
        ordering=False,
    ):
        super().__init__()

        self.depth = depth

        assert learnable_query

        self.ndim = dim
        self.num_inputs = num_inputs
        self.num_latents = num_latents
        self.learnable_query = learnable_query
        self.ordering = ordering

        if self.learnable_query:
            self.query = nn.Parameter(torch.randn(1, num_latents, queries_dim))

        self.cross_attend_blocks = nn.ModuleList(
            [
                PreNorm(
                    dim, Attention(dim, dim, heads=1, dim_head=dim), context_dim=dim
                ),
                PreNorm(dim, FeedForward(dim)),
            ]
        )

        self.point_embed = PointEmbed(dim=dim)

        get_latent_attn = lambda: PreNorm(
            dim, Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1)
        )
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}

        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [get_latent_attn(**cache_args), get_latent_ff(**cache_args)]
                )
            )

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            Attention(queries_dim, dim, heads=1, dim_head=dim),
            context_dim=dim,
        )
        self.decoder_ff = (
            PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        )

        self.to_outputs = (
            nn.Linear(dim, output_dim) if exists(output_dim) else nn.Identity()
        )

    def forward(self, pc):
        # pc: B x N x 3
        B, N, D = pc.shape
        assert N == self.num_inputs

        if not self.learnable_query:
            ###### fps
            flattened = pc.view(B * N, D)

            batch = torch.arange(B).to(pc.device)
            batch = torch.repeat_interleave(batch, N)

            pos = flattened

            ratio = 1.0 * self.num_latents / self.num_inputs

            idx = fps(pos, batch, ratio=ratio)

            sampled_pc = pos[idx]
            sampled_pc = sampled_pc.view(B, -1, 3)

            if self.ordering:
                sampled_pc = torch.sort(sampled_pc, dim=1)[0]
            ######

            sampled_pc_embeddings = self.point_embed(sampled_pc)
        else:
            sampled_pc_embeddings = self.query

        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = (
            cross_attn(sampled_pc_embeddings, context=pc_embeddings, mask=None)
            + sampled_pc_embeddings
        )
        x = cross_ff(x) + x

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        return self.to_outputs(x)


class Decoder(nn.Module):
    def __init__(
        self,
        depth=0,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        dim=512,
        queries_dim=512,
        output_dim=1,
        decoder_ff=False,
    ):
        super().__init__()

        get_latent_attn = lambda: PreNorm(
            dim, Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1)
        )
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}

        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [get_latent_attn(**cache_args), get_latent_ff(**cache_args)]
                )
            )

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            Attention(queries_dim, dim, heads=1, dim_head=dim),
            context_dim=dim,
        )
        self.point_embed = PointEmbed(dim=dim)
        self.decoder_ff = (
            PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        )

        self.to_outputs = (
            nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()
        )

    def forward(self, x, queries):
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context=x)

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        return self.to_outputs(latents)


class KLAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=2048,
        num_latents=512,
        latent_dim=64,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    ):
        super().__init__()

        self.depth = depth

        self.num_inputs = num_inputs
        self.num_latents = num_latents

        self.cross_attend_blocks = nn.ModuleList(
            [
                PreNorm(
                    dim, Attention(dim, dim, heads=1, dim_head=dim), context_dim=dim
                ),
                PreNorm(dim, FeedForward(dim)),
            ]
        )

        self.point_embed = PointEmbed(dim=dim)

        get_latent_attn = lambda: PreNorm(
            dim, Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1)
        )
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}

        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [get_latent_attn(**cache_args), get_latent_ff(**cache_args)]
                )
            )

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            Attention(queries_dim, dim, heads=1, dim_head=dim),
            context_dim=dim,
        )
        self.decoder_ff = (
            PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        )

        self.to_outputs = (
            nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()
        )

        self.proj = nn.Linear(latent_dim, dim)

        self.mean_fc = nn.Linear(dim, latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)

    def encode(self, pc):
        # pc: B x N x 3
        B, N, D = pc.shape
        assert N == self.num_inputs

        ###### fps
        flattened = pc.view(B * N, D)

        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened

        ratio = 1.0 * self.num_latents / self.num_inputs

        idx = fps(pos, batch, ratio=ratio)

        sampled_pc = pos[idx]
        sampled_pc = sampled_pc.view(B, -1, 3)
        ######

        sampled_pc_embeddings = self.point_embed(sampled_pc)

        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = (
            cross_attn(sampled_pc_embeddings, context=pc_embeddings, mask=None)
            + sampled_pc_embeddings
        )
        x = cross_ff(x) + x

        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar)
        x = posterior.sample()
        kl = posterior.kl()

        return kl, x

    def decode(self, x, queries):

        x = self.proj(x)

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # cross attend from decoder queries to latents
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context=x)

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        return self.to_outputs(latents)

    def forward(self, pc, queries):
        kl, x = self.encode(pc)

        o = self.decode(x, queries).squeeze(-1)

        # return o.squeeze(-1), kl
        return {"logits": o, "kl": kl}


class EncoderV1(nn.Module):
    def __init__(
        self,
        *,
        depth=8,
        dim=512,
        queries_dim=3072,
        output_dim=1,
        num_inputs=2048,
        num_latents=512,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        learnable_query=True,
        ordering=False,
    ):
        super().__init__()

        self.depth = depth

        assert learnable_query

        self.ndim = dim
        self.num_inputs = num_inputs
        self.num_latents = num_latents
        self.learnable_query = learnable_query
        self.ordering = ordering

        if self.learnable_query:
            self.query = nn.Parameter(torch.randn(1, num_latents, queries_dim))

        self.cross_attend_blocks = nn.ModuleList(
            [
                PreNorm(
                    dim, Attention(dim, dim, heads=1, dim_head=dim), context_dim=dim
                ),
                PreNorm(dim, FeedForward(dim)),
            ]
        )

        self.point_embed = PointEmbed(dim=dim)

        get_latent_attn = lambda: PreNorm(
            dim, Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1)
        )
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_cross_attn = lambda: PreNorm(
            dim, Attention(dim, dim, heads=1, dim_head=dim), context_dim=dim
        )
        get_latent_attn, get_latent_ff, get_latent_cross_attn = map(
            cache_fn, (get_latent_attn, get_latent_ff, get_latent_cross_attn)
        )

        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}

        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        get_latent_attn(**cache_args),
                        get_latent_cross_attn(**cache_args),
                        get_latent_ff(**cache_args),
                    ]
                )
            )

        self.to_outputs = (
            nn.Linear(dim, output_dim) if exists(output_dim) else nn.Identity()
        )

    def forward(self, pc):
        # pc: B x N x 3
        B, N, D = pc.shape
        assert N == self.num_inputs

        if not self.learnable_query:
            ###### fps
            flattened = pc.view(B * N, D)

            batch = torch.arange(B).to(pc.device)
            batch = torch.repeat_interleave(batch, N)

            pos = flattened

            ratio = 1.0 * self.num_latents / self.num_inputs

            idx = fps(pos, batch, ratio=ratio)

            sampled_pc = pos[idx]
            sampled_pc = sampled_pc.view(B, -1, 3)

            if self.ordering:
                sampled_pc = torch.sort(sampled_pc, dim=1)[0]
            ######

            sampled_pc_embeddings = self.point_embed(sampled_pc)
        else:
            sampled_pc_embeddings = self.query

        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = (
            cross_attn(sampled_pc_embeddings, context=pc_embeddings, mask=None)
            + sampled_pc_embeddings
        )
        x = cross_ff(x) + x

        for self_attn, cross_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = cross_attn(x, context=pc_embeddings) + x
            x = self_ff(x) + x

        return self.to_outputs(x)
