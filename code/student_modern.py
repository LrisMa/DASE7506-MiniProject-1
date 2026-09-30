"""Modern causal Transformer with GQA, RoPE, RMSNorm, and SwiGLU."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, epsilon=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.epsilon).to(x.dtype) * self.weight


def rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class ModernBlock(nn.Module):
    def __init__(self, width, heads, kv_heads, hidden, dropout, context):
        super().__init__()
        if width % heads or heads % kv_heads or (width // heads) % 2:
            raise ValueError('Head dimensions must be even, with width divisible by heads and heads by KV heads.')
        self.heads, self.kv_heads = heads, kv_heads
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, kv_heads * (width // heads), bias=False)
        self.value = nn.Linear(width, kv_heads * (width // heads), bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.gate_value = nn.Linear(width, 2 * hidden, bias=False)
        self.output = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)
        head_width = width // heads
        positions = torch.arange(context, dtype=torch.float32)
        frequencies = 1. / (10000. ** (torch.arange(0, head_width, 2, dtype=torch.float32) / head_width))
        angles = torch.outer(positions, frequencies)
        self.register_buffer('rope_cos', angles.cos().repeat_interleave(2, dim=-1), persistent=False)
        self.register_buffer('rope_sin', angles.sin().repeat_interleave(2, dim=-1), persistent=False)

    def apply_rope(self, x):
        length = x.shape[2]
        cosine = self.rope_cos[:length].to(x.dtype)
        sine = self.rope_sin[:length].to(x.dtype)
        return x * cosine.unsqueeze(0).unsqueeze(0) + rotate_half(x) * sine.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        batch, length, width = x.shape
        head_width = width // self.heads
        query = self.query(self.norm1(x)).view(batch, length, self.heads, head_width).transpose(1, 2)
        key = self.key(self.norm1(x)).view(batch, length, self.kv_heads, head_width).transpose(1, 2)
        value = self.value(self.norm1(x)).view(batch, length, self.kv_heads, head_width).transpose(1, 2)
        query, key = self.apply_rope(query), self.apply_rope(key)
        repeats = self.heads // self.kv_heads
        key, value = key.repeat_interleave(repeats, dim=1), value.repeat_interleave(repeats, dim=1)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True,
                              dropout_p=self.dropout.p if self.training else 0.)
        x = x + self.dropout(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        gate, value = self.gate_value(self.norm2(x)).chunk(2, dim=-1)
        return x + self.dropout(self.output(F.silu(gate) * value))


class ModernGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context = config['context']
        width = config['width']
        heads = config['heads']
        hidden = config.get('hidden', (8 * width) // 3)
        dropout = float(config.get('dropout', 0.))
        self.token = nn.Embedding(config['vocab'], width)
        self.blocks = nn.ModuleList([
            ModernBlock(width, heads, config.get('kv_heads', heads), hidden, dropout, self.context)
            for _ in range(config['depth'])
        ])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight
        residual_std = .02 / math.sqrt(2 * config['depth'])
        for block in self.blocks:
            nn.init.normal_(block.proj.weight, std=residual_std)
            nn.init.normal_(block.output.weight, std=residual_std)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)

    def forward(self, ids):
        x = self.token(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return ModernGPT(config)