"""Create an evaluation checkpoint with a selected train-derived trigram mixture."""
import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--alpha', required=True, type=float)
    parser.add_argument('--asset', default='assets/trigram_top4.pt')
    args = parser.parse_args()
    if not 0. <= args.alpha < 1.:
        parser.error('alpha must be in [0, 1).')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = dict(checkpoint['config'])
    config.update(trigram_asset=args.asset, trigram_alpha=args.alpha)
    checkpoint['config'] = config
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)


if __name__ == '__main__':
    main()