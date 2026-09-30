"""Student model: SwiGLU GPT with an optional train-only trigram mixture."""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SwiGLUBlock(nn.Module):
    def __init__(self, width=128, heads=4, dropout=0.):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        hidden = (8 * width) // 3
        self.gate_value = nn.Linear(width, 2 * hidden)
        self.output = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.dropout(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        gate, value = self.gate_value(self.norm2(x)).chunk(2, dim=-1)
        return x + self.dropout(self.output(F.silu(gate) * value))


class TrigramMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        asset = config.get('trigram_asset')
        self.alpha = float(config.get('trigram_alpha', 0.))
        self.tau = float(config.get('trigram_tau', 0.))
        self.coverage_weighting = bool(config.get('trigram_coverage_weighting', False))
        if not 0. <= self.alpha < 1.:
            raise ValueError('trigram_alpha must be in [0, 1).')
        if self.tau < 0.:
            raise ValueError('trigram_tau must be non-negative.')
        self.enabled = asset is not None and self.alpha > 0.
        if not self.enabled:
            return
        state = torch.load(Path(__file__).resolve().parent / asset, map_location='cpu', weights_only=True)
        if state['vocab'] != config['vocab']:
            raise ValueError('Trigram asset vocabulary does not match the model configuration.')
        self.register_buffer('lookup', state['lookup'], persistent=False)
        self.register_buffer('next_tokens', state['next_tokens'], persistent=False)
        self.register_buffer('probabilities', state['probabilities'], persistent=False)
        if self.tau > 0.:
            if 'context_counts' not in state:
                raise ValueError('Adaptive trigram mixing requires an asset with context counts.')
            self.register_buffer('context_counts', state['context_counts'], persistent=False)

    def forward(self, logp, ids):
        if not self.enabled or ids.shape[1] < 2:
            return logp
        vocabulary = logp.shape[-1]
        pairs = ids[:, :-1] * vocabulary + ids[:, 1:]
        rows = self.lookup[pairs].to(torch.long)
        safe_rows = rows.clamp_min(0)
        next_tokens = self.next_tokens[safe_rows].to(torch.long)
        probabilities = self.probabilities[safe_rows].float()
        coverage = probabilities.sum(-1).clamp_max(1.)
        probabilities = probabilities / coverage.unsqueeze(-1).clamp_min(torch.finfo(torch.float32).tiny)
        valid = (rows >= 0) & (coverage > 0)
        if self.tau > 0.:
            counts = self.context_counts[safe_rows].float()
            alpha = self.alpha * counts / (counts + self.tau)
        else:
            alpha = torch.full_like(rows, self.alpha, dtype=torch.float32)
        if self.coverage_weighting:
            alpha = alpha * coverage
        alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
        log_neural = logp[:, 1:, :]
        log_neural_weight = torch.log1p(-alpha)
        mixed = log_neural + log_neural_weight.unsqueeze(-1)
        neural_at_tokens = log_neural.gather(-1, next_tokens)
        ngram_at_tokens = probabilities.clamp_min(torch.finfo(torch.float32).tiny).log() + torch.log(alpha.clamp_min(torch.finfo(torch.float32).tiny)).unsqueeze(-1)
        replacement = torch.logaddexp(neural_at_tokens + log_neural_weight.unsqueeze(-1), ngram_at_tokens)
        active = valid.unsqueeze(-1) & (probabilities > 0) & (alpha.unsqueeze(-1) > 0)
        existing = mixed.gather(-1, next_tokens)
        mixed.scatter_(-1, next_tokens, torch.where(active, replacement, existing))
        result = logp.clone()
        result[:, 1:, :] = mixed
        return result


class FourgramMixer(nn.Module):
    """Interpolate a sparse train-only fourgram table after lower-order experts."""
    def __init__(self, config):
        super().__init__()
        asset = config.get('fourgram_asset')
        self.alpha = float(config.get('fourgram_alpha', 0.))
        if not 0. <= self.alpha < 1.:
            raise ValueError('fourgram_alpha must be in [0, 1).')
        self.enabled = asset is not None and self.alpha > 0.
        if not self.enabled:
            return
        state = torch.load(Path(__file__).resolve().parent / asset, map_location='cpu', weights_only=True)
        if state['vocab'] != config['vocab']:
            raise ValueError('Fourgram asset vocabulary does not match the model configuration.')
        self.register_buffer('contexts', state['contexts'], persistent=False)
        self.register_buffer('next_tokens', state['next_tokens'], persistent=False)
        self.register_buffer('probabilities', state['probabilities'], persistent=False)

    def forward(self, logp, ids):
        if not self.enabled or ids.shape[1] < 3:
            return logp
        vocabulary = logp.shape[-1]
        context = ((ids[:, :-2] * vocabulary + ids[:, 1:-1]) * vocabulary + ids[:, 2:]).to(torch.long)
        rows = torch.searchsorted(self.contexts, context)
        safe_rows = rows.clamp_max(len(self.contexts) - 1)
        valid = (rows < len(self.contexts)) & (self.contexts[safe_rows] == context)
        next_tokens = self.next_tokens[safe_rows].to(torch.long)
        alpha = self.alpha * self.probabilities[safe_rows].float()
        alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
        original = logp[:, 2:, :]
        original_weight = torch.log1p(-alpha)
        mixed = original + original_weight.unsqueeze(-1)
        existing = mixed.gather(-1, next_tokens.unsqueeze(-1))
        replacement = torch.logaddexp(
            original.gather(-1, next_tokens.unsqueeze(-1)) + original_weight.unsqueeze(-1),
            torch.log(alpha.clamp_min(torch.finfo(torch.float32).tiny)).unsqueeze(-1)
        )
        mixed.scatter_(-1, next_tokens.unsqueeze(-1), torch.where(valid.unsqueeze(-1), replacement, existing))
        result = logp.clone()
        result[:, 2:, :] = mixed
        return result


class FivegramMixer(nn.Module):
    """Interpolate a sparse train-only fivegram table after lower-order experts."""
    def __init__(self, config):
        super().__init__()
        asset = config.get('fivegram_asset')
        self.alpha = float(config.get('fivegram_alpha', 0.))
        if not 0. <= self.alpha < 1.:
            raise ValueError('fivegram_alpha must be in [0, 1).')
        self.enabled = asset is not None and self.alpha > 0.
        if not self.enabled:
            return
        state = torch.load(Path(__file__).resolve().parent / asset, map_location='cpu', weights_only=True)
        if state['vocab'] != config['vocab']:
            raise ValueError('Fivegram asset vocabulary does not match the model configuration.')
        self.register_buffer('contexts', state['contexts'], persistent=False)
        self.register_buffer('next_tokens', state['next_tokens'], persistent=False)
        self.register_buffer('probabilities', state['probabilities'], persistent=False)

    def forward(self, logp, ids):
        if not self.enabled or ids.shape[1] < 4:
            return logp
        vocabulary = logp.shape[-1]
        context = (((ids[:, :-3] * vocabulary + ids[:, 1:-2]) * vocabulary + ids[:, 2:-1]) * vocabulary + ids[:, 3:]).to(torch.long)
        rows = torch.searchsorted(self.contexts, context)
        safe_rows = rows.clamp_max(len(self.contexts) - 1)
        valid = (rows < len(self.contexts)) & (self.contexts[safe_rows] == context)
        next_tokens = self.next_tokens[safe_rows].to(torch.long)
        alpha = self.alpha * self.probabilities[safe_rows].float()
        alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
        original = logp[:, 3:, :]
        original_weight = torch.log1p(-alpha)
        mixed = original + original_weight.unsqueeze(-1)
        existing = mixed.gather(-1, next_tokens.unsqueeze(-1))
        replacement = torch.logaddexp(
            original.gather(-1, next_tokens.unsqueeze(-1)) + original_weight.unsqueeze(-1),
            torch.log(alpha.clamp_min(torch.finfo(torch.float32).tiny)).unsqueeze(-1)
        )
        mixed.scatter_(-1, next_tokens.unsqueeze(-1), torch.where(valid.unsqueeze(-1), replacement, existing))
        result = logp.clone()
        result[:, 3:, :] = mixed
        return result


class SwiGLUGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        dropout = float(config.get('dropout', 0.))
        self.dropout = nn.Dropout(dropout)
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = nn.Embedding(self.context, width)
        self.blocks = nn.ModuleList([SwiGLUBlock(width, config['heads'], dropout) for _ in range(config['depth'])])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.trigram = TrigramMixer(config)
        self.fourgram = FourgramMixer(config)
        self.fivegram = FivegramMixer(config)
        self.cache_alpha = float(config.get('cache_alpha', 0.))
        self.cache_temperature = float(config.get('cache_temperature', .1))
        self.cache_confidence_gating = bool(config.get('cache_confidence_gating', False))
        if not 0. <= self.cache_alpha < 1.:
            raise ValueError('cache_alpha must be in [0, 1).')
        if self.cache_temperature <= 0.:
            raise ValueError('cache_temperature must be positive.')
        peer_config = config.get('ensemble_peer_config')
        self.peer_weight = float(config.get('ensemble_peer_weight', 0.))
        self.ensemble_geometric = bool(config.get('ensemble_geometric', False))
        if not 0. <= self.peer_weight <= 1.:
            raise ValueError('ensemble_peer_weight must be in [0, 1].')
        self.peer_cache_alpha = float(config.get('ensemble_peer_cache_alpha', 0.))
        self.peer_cache_temperature = float(config.get('ensemble_peer_cache_temperature', .075))
        if not 0. <= self.peer_cache_alpha < 1.:
            raise ValueError('ensemble_peer_cache_alpha must be in [0, 1).')
        if self.peer_cache_temperature <= 0.:
            raise ValueError('ensemble_peer_cache_temperature must be positive.')
        self.peer = SwiGLUGPT(peer_config) if peer_config is not None else None
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.dropout(self.token(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device)))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def cache_log_probs(self, features, ids, temperature=None, return_confidence=False):
        """Return a causal, call-local distribution over continuations seen earlier in this window."""
        batch, length, width = features.shape
        vocabulary = self.head.out_features
        normalized = F.normalize(features.float(), dim=-1)
        temperature = self.cache_temperature if temperature is None else temperature
        scores = normalized @ normalized.transpose(-2, -1) / temperature
        future_or_current = torch.ones(length, length, dtype=torch.bool, device=ids.device).triu()
        weights = torch.softmax(scores.masked_fill(future_or_current, -torch.inf), dim=-1)
        weights = torch.nan_to_num(weights)
        probabilities = features.new_zeros(batch, length, vocabulary, dtype=torch.float32)
        targets = ids[:, 1:].unsqueeze(1).expand(-1, length, -1)
        probabilities.scatter_add_(2, targets, weights[:, :, :-1])
        logp = probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
        if not return_confidence:
            return logp
        history = torch.arange(length, device=ids.device, dtype=torch.float32)
        uniform_peak = history.reciprocal().masked_fill(history == 0., 0.)
        confidence = (weights.amax(dim=-1) - uniform_peak) / (1. - uniform_peak).clamp_min(torch.finfo(torch.float32).tiny)
        confidence = confidence.clamp(0., 1.) * (history > 1).to(confidence.dtype)
        return logp, confidence

    def predict_log_probs(self, ids):
        features = self.features(ids)
        logp = F.log_softmax(self.head(features).float(), dim=-1)
        if self.peer is not None and self.peer_weight > 0.:
            peer_features = self.peer.features(ids)
            peer_logp = F.log_softmax(self.peer.head(peer_features).float(), dim=-1)
            if self.peer_cache_alpha > 0. and ids.shape[1] > 1:
                peer_cache_logp = self.peer.cache_log_probs(
                    peer_features, ids, self.peer_cache_temperature
                )
                peer_alpha = torch.full_like(peer_logp[:, :, 0], self.peer_cache_alpha)
                peer_alpha[:, 0] = 0.
                peer_logp = torch.logaddexp(
                    peer_logp + torch.log1p(-peer_alpha).unsqueeze(-1),
                    peer_cache_logp + torch.log(peer_alpha.clamp_min(torch.finfo(torch.float32).tiny)).unsqueeze(-1)
                )
            if self.ensemble_geometric:
                combined = (1. - self.peer_weight) * logp + self.peer_weight * peer_logp
                logp = F.log_softmax(combined, dim=-1)
            else:
                logp = torch.logaddexp(logp + math.log1p(-self.peer_weight),
                                       peer_logp + math.log(self.peer_weight))
        if self.cache_alpha > 0. and ids.shape[1] > 1:
            if self.cache_confidence_gating:
                cache_logp, confidence = self.cache_log_probs(features, ids, return_confidence=True)
            else:
                cache_logp = self.cache_log_probs(features, ids)
            alpha = torch.full_like(logp[:, :, 0], self.cache_alpha)
            if self.cache_confidence_gating:
                alpha = alpha * confidence
            alpha[:, 0] = 0.
            logp = torch.logaddexp(logp + torch.log1p(-alpha).unsqueeze(-1),
                                   cache_logp + torch.log(alpha.clamp_min(torch.finfo(torch.float32).tiny)).unsqueeze(-1))
        return self.fivegram(self.fourgram(self.trigram(logp, ids), ids), ids)


def build_model(config):
    return SwiGLUGPT(config)


def build_trigram_asset(output_path, top_k=4):
    """Build a compact top-k trigram continuation table from the training split only."""
    from common import load_data

    if top_k < 1:
        raise ValueError('top_k must be positive.')
    tokens, _ = load_data()['train']
    tokens = tokens.numpy().astype(np.int64, copy=False)
    vocab = 2048
    codes = ((tokens[:-2] * vocab + tokens[1:-1]) * vocab + tokens[2:]).copy()
    codes.sort()
    starts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1]
    unique_codes = codes[starts]
    counts = np.diff(np.r_[starts, len(codes)]).astype(np.float32)
    del codes

    contexts, continuations = divmod(unique_codes, vocab)
    order = np.lexsort((continuations, -counts, contexts))
    ordered_contexts = contexts[order]
    group_start = np.r_[True, ordered_contexts[1:] != ordered_contexts[:-1]]
    group_ids = np.cumsum(group_start, dtype=np.int32) - 1
    group_positions = np.arange(len(order), dtype=np.int32)
    first_positions = np.maximum.accumulate(np.where(group_start, group_positions, 0))
    ranks = group_positions - first_positions
    totals = np.add.reduceat(counts[order], np.flatnonzero(group_start))
    selected = ranks < top_k
    selected_groups = group_ids[selected]

    lookup = np.full(vocab * vocab, -1, dtype=np.int32)
    lookup[ordered_contexts[group_start]] = np.arange(len(totals), dtype=np.int32)
    next_tokens = np.zeros((len(totals), top_k), dtype=np.int16)
    probabilities = np.zeros((len(totals), top_k), dtype=np.float16)
    next_tokens[selected_groups, ranks[selected]] = continuations[order][selected]
    probabilities[selected_groups, ranks[selected]] = (counts[order][selected] / totals[selected_groups]).astype(np.float16)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'vocab': vocab, 'lookup': torch.from_numpy(lookup),
                'next_tokens': torch.from_numpy(next_tokens),
                'probabilities': torch.from_numpy(probabilities),
                'context_counts': torch.from_numpy(totals.astype(np.int32))}, output_path)
    asset_bytes = output_path.stat().st_size
    if asset_bytes > 64 * 1024 * 1024:
        output_path.unlink()
        raise ValueError(f'Trigram asset is {asset_bytes / 2**20:.1f} MiB, above the 64 MiB limit.')
    print(f'Built {len(totals):,} trigram contexts; asset size {asset_bytes / 2**20:.1f} MiB.')


def build_fourgram_asset(output_path, min_count=3):
    """Build a sparse train-only top-1 fourgram table for frequent three-token contexts."""
    from common import load_data

    if min_count < 1:
        raise ValueError('min_count must be positive.')
    tokens, _ = load_data()['train']
    tokens = tokens.numpy().astype(np.int64, copy=False)
    vocab = 2048
    codes = ((((tokens[:-3] * vocab + tokens[1:-2]) * vocab + tokens[2:-1]) * vocab) + tokens[3:]).copy()
    codes.sort()
    starts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1]
    unique_codes = codes[starts]
    counts = np.diff(np.r_[starts, len(codes)]).astype(np.float32)
    del codes

    contexts, continuations = divmod(unique_codes, vocab)
    order = np.lexsort((continuations, -counts, contexts))
    ordered_contexts = contexts[order]
    group_start = np.r_[True, ordered_contexts[1:] != ordered_contexts[:-1]]
    group_ids = np.cumsum(group_start, dtype=np.int32) - 1
    totals = np.add.reduceat(counts[order], np.flatnonzero(group_start))
    ranks = np.arange(len(order), dtype=np.int32) - np.maximum.accumulate(
        np.where(group_start, np.arange(len(order), dtype=np.int32), 0)
    )
    selected = (ranks == 0) & (totals[group_ids] >= min_count)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'vocab': vocab, 'contexts': torch.from_numpy(ordered_contexts[selected]),
                'next_tokens': torch.from_numpy(continuations[order][selected].astype(np.int16)),
                'probabilities': torch.from_numpy((counts[order][selected] / totals[group_ids[selected]]).astype(np.float16))},
               output_path)
    asset_bytes = output_path.stat().st_size
    print(f'Built {selected.sum():,} frequent fourgram contexts; asset size {asset_bytes / 2**20:.1f} MiB.')


def build_fivegram_asset(output_path, min_count=3):
    """Build a sparse train-only top-1 fivegram table for frequent four-token contexts."""
    from common import load_data

    if min_count < 1:
        raise ValueError('min_count must be positive.')
    tokens, _ = load_data()['train']
    tokens = tokens.numpy().astype(np.int64, copy=False)
    vocab = 2048
    codes = (((((tokens[:-4] * vocab + tokens[1:-3]) * vocab + tokens[2:-2]) * vocab + tokens[3:-1]) * vocab) + tokens[4:]).copy()
    codes.sort()
    starts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1]
    unique_codes = codes[starts]
    counts = np.diff(np.r_[starts, len(codes)]).astype(np.float32)
    del codes

    contexts, continuations = divmod(unique_codes, vocab)
    order = np.lexsort((continuations, -counts, contexts))
    ordered_contexts = contexts[order]
    group_start = np.r_[True, ordered_contexts[1:] != ordered_contexts[:-1]]
    group_ids = np.cumsum(group_start, dtype=np.int32) - 1
    totals = np.add.reduceat(counts[order], np.flatnonzero(group_start))
    positions = np.arange(len(order), dtype=np.int32)
    ranks = positions - np.maximum.accumulate(np.where(group_start, positions, 0))
    selected = (ranks == 0) & (totals[group_ids] >= min_count)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'vocab': vocab, 'contexts': torch.from_numpy(ordered_contexts[selected]),
                'next_tokens': torch.from_numpy(continuations[order][selected].astype(np.int16)),
                'probabilities': torch.from_numpy((counts[order][selected] / totals[group_ids[selected]]).astype(np.float16))},
               output_path)
    asset_bytes = output_path.stat().st_size
    print(f'Built {selected.sum():,} frequent fivegram contexts; asset size {asset_bytes / 2**20:.1f} MiB.')


def make_trigram_checkpoint(source_path, output_path, asset, alpha, tau=0., coverage_weighting=False,
                            peer_checkpoint=None, peer_weight=0., cache_alpha=0., cache_temperature=.1,
                            peer_cache_alpha=0., peer_cache_temperature=.075, cache_confidence_gating=False,
                            fourgram_asset=None, fourgram_alpha=0., fivegram_asset=None, fivegram_alpha=0.,
                            ensemble_geometric=False):
    """Reuse frozen neural weights while adding a train-derived trigram asset to its config."""
    if not 0. <= alpha < 1.:
        raise ValueError('alpha must be in [0, 1).')
    source_path, output_path = Path(source_path), Path(output_path)
    checkpoint = torch.load(source_path, map_location='cpu', weights_only=True)
    checkpoint['config'] = dict(checkpoint['config'], trigram_asset=asset,
                                trigram_alpha=alpha, trigram_tau=tau,
                                trigram_coverage_weighting=coverage_weighting,
                                cache_alpha=cache_alpha, cache_temperature=cache_temperature,
                                cache_confidence_gating=cache_confidence_gating,
                                fourgram_asset=fourgram_asset, fourgram_alpha=fourgram_alpha,
                                fivegram_asset=fivegram_asset, fivegram_alpha=fivegram_alpha,
                                ensemble_geometric=ensemble_geometric,
                                ensemble_peer_cache_alpha=peer_cache_alpha,
                                ensemble_peer_cache_temperature=peer_cache_temperature)
    if peer_checkpoint is not None:
        if not 0. < peer_weight < 1.:
            raise ValueError('ensemble peer weight must be in (0, 1).')
        peer = torch.load(peer_checkpoint, map_location='cpu', weights_only=True)
        checkpoint['config']['ensemble_peer_config'] = peer['config']
        checkpoint['config']['ensemble_peer_weight'] = peer_weight
        checkpoint['model'].update({f'peer.{name}': value for name, value in peer['model'].items()})
        checkpoint['ensemble_peer_checkpoint'] = str(peer_checkpoint)
        checkpoint['ensemble_peer_weight'] = peer_weight
    checkpoint['trigram_source_checkpoint'] = str(source_path)
    checkpoint['trigram_alpha'] = alpha
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-trigram-asset', action='store_true')
    parser.add_argument('--build-fourgram-asset', action='store_true')
    parser.add_argument('--build-fivegram-asset', action='store_true')
    parser.add_argument('--asset', default='assets/trigram_top4.pt')
    parser.add_argument('--fourgram-asset', default='assets/fourgram_top1_min3.pt')
    parser.add_argument('--fivegram-asset', default='assets/fivegram_top1_min3.pt')
    parser.add_argument('--fourgram-alpha', type=float, default=0.)
    parser.add_argument('--fivegram-alpha', type=float, default=0.)
    parser.add_argument('--top-k', type=int, default=4)
    parser.add_argument('--fourgram-min-count', type=int, default=3)
    parser.add_argument('--fivegram-min-count', type=int, default=3)
    parser.add_argument('--source-checkpoint', type=Path)
    parser.add_argument('--output-checkpoint', type=Path)
    parser.add_argument('--trigram-alpha', type=float)
    parser.add_argument('--trigram-tau', type=float, default=0.)
    parser.add_argument('--trigram-coverage-weighting', action='store_true')
    parser.add_argument('--ensemble-peer-checkpoint', type=Path)
    parser.add_argument('--ensemble-peer-weight', type=float, default=0.)
    parser.add_argument('--ensemble-geometric', action='store_true')
    parser.add_argument('--cache-alpha', type=float, default=0.)
    parser.add_argument('--cache-temperature', type=float, default=.1)
    parser.add_argument('--cache-confidence-gating', action='store_true')
    parser.add_argument('--ensemble-peer-cache-alpha', type=float, default=0.)
    parser.add_argument('--ensemble-peer-cache-temperature', type=float, default=.075)
    args = parser.parse_args()
    if args.build_trigram_asset:
        build_trigram_asset(args.asset, args.top_k)
    if args.build_fourgram_asset:
        build_fourgram_asset(args.fourgram_asset, args.fourgram_min_count)
    if args.build_fivegram_asset:
        build_fivegram_asset(args.fivegram_asset, args.fivegram_min_count)
    if args.source_checkpoint is not None:
        if args.output_checkpoint is None or args.trigram_alpha is None:
            parser.error('--source-checkpoint requires --output-checkpoint and --trigram-alpha.')
        make_trigram_checkpoint(args.source_checkpoint, args.output_checkpoint,
                                args.asset, args.trigram_alpha, args.trigram_tau,
                                args.trigram_coverage_weighting,
                                args.ensemble_peer_checkpoint, args.ensemble_peer_weight,
                                args.cache_alpha, args.cache_temperature,
                                args.ensemble_peer_cache_alpha, args.ensemble_peer_cache_temperature,
                                args.cache_confidence_gating,
                                args.fourgram_asset, args.fourgram_alpha,
                                args.fivegram_asset, args.fivegram_alpha,
                                args.ensemble_geometric)


if __name__ == '__main__':
    main()
