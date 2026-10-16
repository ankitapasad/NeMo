#!/usr/bin/env python3
"""Estimate context-aware duration bins from one native training input_cfg source."""
import argparse
import json
from pathlib import Path

from nemo.collections.speechlm2.data.batch_estimation import estimate_context_duration_bins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-cfg', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-buckets', type=int, default=10)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a fresh path')
    result = estimate_context_duration_bins(args.input_cfg, seed=args.seed, num_buckets=args.num_buckets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
