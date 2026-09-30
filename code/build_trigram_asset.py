"""Build a compact, train-only top-k trigram asset for the student model."""
import argparse
from pathlib import Path
import numpy as np
import torch
from common import ROOT, load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'assets/trigram_top4.pt')
    parser.add_argument('--top-k', type=int, default=4)
    parser.add_argument('--min-count', type=int, default=3)
    args = parser.parse_args()
    if args.top_k < 1 or args.min_count < 1:
        parser.error('top-k and min-count must be positive.')
    tokens = load_data()['train'][0].numpy().astype(np.uint64, copy=False)
    vocabulary = 2048
    triples = (tokens[:-2] * vocabulary + tokens[1:-1]) * vocabulary + tokens[2:]
    codes, counts = np.unique(triples, return_counts=True)
    pairs, continuations = divmod(codes, vocabulary)
    starts = np.r_[0, np.flatnonzero(np.diff(pairs)) + 1]
    ends = np.r_[starts[1:], len(pairs)]
    totals = np.add.reduceat(counts, starts)
    keep = totals >= args.min_count
    selected_starts, selected_ends = starts[keep], ends[keep]
    selected_pairs = pairs[selected_starts].astype(np.int64, copy=False)
    lookup = np.full(vocabulary * vocabulary, -1, dtype=np.int32)
    lookup[selected_pairs] = np.arange(len(selected_pairs), dtype=np.int32)
    next_tokens = np.zeros((len(selected_pairs), args.top_k), dtype=np.int16)
    probabilities = np.zeros((len(selected_pairs), args.top_k), dtype=np.float16)
    for row, (start, end) in enumerate(zip(selected_starts, selected_ends)):
        group_counts = counts[start:end]
        if len(group_counts) > args.top_k:
            chosen = np.argpartition(group_counts, -args.top_k)[-args.top_k:]
        else:
            chosen = np.arange(len(group_counts))
        chosen_counts = group_counts[chosen]
        size = len(chosen)
        next_tokens[row, :size] = continuations[start:end][chosen]
        probabilities[row, :size] = chosen_counts / chosen_counts.sum()
    state = {
        'vocab': vocabulary,
        'lookup': torch.from_numpy(lookup),
        'next_tokens': torch.from_numpy(next_tokens),
        'probabilities': torch.from_numpy(probabilities),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    size = sum(tensor.numel() * tensor.element_size() for tensor in state.values() if isinstance(tensor, torch.Tensor))
    print(f'contexts={len(selected_pairs)} asset_bytes={size} output={args.output}')


if __name__ == '__main__':
    main()