#!/usr/bin/env python3
"""Run prepare/profile/finalize stages; see batch_pipeline --help and the plan YAML."""
from nemo.collections.speechlm2.data.batch_pipeline import main

if __name__ == '__main__':
    raise SystemExit(main())
