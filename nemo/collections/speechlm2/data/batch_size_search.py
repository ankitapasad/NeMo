# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone copy of the legacy speechlm2 OOMptimizer search algorithm.

The original scripts/speechlm2/oomptimizer.py remains unchanged and does not
import this class. Only the real-data connector uses this copy. Parity tests
compare search transitions against the legacy implementation. Caller-selected
starting sizes and caps may differ; doubling and interval narrowing are retained.
Batch construction, GPU execution, memory-budget enforcement and subprocess
isolation are caller responsibilities. The connector defaults to an 80% memory
budget; the untouched original CLI retains its 90% default.
"""


class BatchSizeSearch:
    def __init__(self, start_batch_size=32, rel_gap_thresh=.05):
        if type(start_batch_size) is not int or start_batch_size < 1:
            raise ValueError('start_batch_size must be a positive integer')
        if not 0 <= rel_gap_thresh <= 1:
            raise ValueError('rel_gap_thresh must be between zero and one')
        self.start_batch_size = start_batch_size
        self.rel_gap_thresh = rel_gap_thresh
        self.reset()

    @property
    def current_batch_size(self):
        return self._current

    @property
    def max_batch_size(self) -> int | None:
        """
        Return the solution of the batch size search problem.
        It will keep returning None until the search is done.
        """
        if (
            self._max_ok is not None
            and self._min_err is not None
            and (self.current_rel_gap <= self.rel_gap_thresh or self._min_err - self._max_ok <= 1)
        ):
            return self._max_ok
        return None

    @property
    def current_rel_gap(self) -> float | None:
        """
        Return the current gap between the largest batch that works and the smallest batch that triggers OOM.
        The gap is defined as the batch size difference divided by the larger element.
        E.g., if the best found batch size is 95 and the smallest that triggers OOM is 100, the gap is 0.05.
        """
        if self._min_err is None or self._max_ok is None:
            return None
        return (self._min_err - self._max_ok) / self._min_err

    def reset(self):
        """Reset the generator to prepare it for a new search."""
        self._current = self.start_batch_size
        self._max_ok = None  # max batch size that works
        self._min_err = None  # min batch size that doesn't work

    def advance(self, oom: bool) -> bool:
        """
        Adjusts the current batch size based on the outcome.
        Returns a bool indicating whether the calibration is complete.
        """
        if self.max_batch_size is not None:
            return True

        if oom:
            # Training step failed with OOM.
            # Update the minimum known batch size that causes an error.
            self._min_err = min(float("inf") if self._min_err is None else self._min_err, self._current)
            # Training step failed on OOM
            if self._max_ok is None:
                # We haven't found a batch size that works yet, keep going 2x down.
                self._current = round(self._current / 2)
            else:
                # Try the middle-point between the known extremes.
                self._current = round((self._max_ok + self._min_err) / 2)
        else:
            # Training step successful.
            # Update the maximum known batch size that works.
            self._max_ok = max(-1 if self._max_ok is None else self._max_ok, self._current)
            if self._min_err is None:
                # We haven't found a batch size that causes an error yet, keep going 2x higher
                self._current *= 2
            else:
                # Try the middle-point between the known extremes.
                self._current = round((self._max_ok + self._min_err) / 2)

        return False
