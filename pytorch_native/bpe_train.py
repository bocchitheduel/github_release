"""Byte-level GPT-2 BPE training (requires ``regex``; CUDA optionally uses torch).

Example: python bpe_train.py corpus.txt --vocab-size 10000 --parallel
"""
from __future__ import annotations

import argparse
import heapq
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import multiprocessing
import os
from pathlib import Path
from typing import Iterable, Iterator

import regex

GPT2_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
_PRETOKEN = regex.compile(GPT2_PATTERN)
_CONTRACTION_PREFIXES = ("'", "'r", "'v", "'l")


@dataclass(frozen=True)
class _PairCandidate:
    pair: tuple[int, int]
    priority: tuple[int, bytes, bytes]
    version: int

    def __lt__(self, other: _PairCandidate) -> bool:
        # Reverse comparison gives heapq a maximum-priority root.
        return (self.priority, self.version) > (other.priority, other.version)


class _PairHeap:
    """Maximum-priority candidates with lazy removal of outdated versions."""

    def __init__(self, counts: Counter, vocab: dict[int, bytes]):
        self.counts = counts
        self.vocab = vocab
        self.versions = {pair: 0 for pair in counts}
        self.heap = [self._entry(pair) for pair in counts]
        heapq.heapify(self.heap)

    def _entry(self, pair: tuple[int, int]) -> _PairCandidate:
        return _PairCandidate(pair, (self.counts[pair], self.vocab[pair[0]],
                                    self.vocab[pair[1]]), self.versions[pair])

    def refresh(self, pair: tuple[int, int]) -> None:
        # Keep versions even after a count becomes zero: if this pair reappears,
        # an earlier entry with the same frequency must remain invalid.
        self.versions[pair] = self.versions.get(pair, 0) + 1
        if self.counts[pair] > 0:
            heapq.heappush(self.heap, self._entry(pair))

    def pop(self) -> tuple[int, int]:
        while self.heap:
            candidate = heapq.heappop(self.heap)
            if (candidate.version == self.versions[candidate.pair]
                    and self.counts[candidate.pair] > 0
                    and candidate.priority[0] == self.counts[candidate.pair]):
                return candidate.pair
        raise RuntimeError("no valid BPE candidate remains")

    def compact(self) -> None:
        # Bound obsolete entries without a full scan on every merge.
        if len(self.heap) > 4 * len(self.counts) + 1024:
            self.heap = [self._entry(pair) for pair in self.counts]
            heapq.heapify(self.heap)


@dataclass
class BPETrainingResult:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: dict[str, int]
    device: str

    def save(self, path: str | os.PathLike) -> None:
        """Lossless JSON: byte strings are represented as hexadecimal."""
        Path(path).write_text(json.dumps({
            "vocab": {str(i): value.hex() for i, value in self.vocab.items()},
            "merges": [[a.hex(), b.hex()] for a, b in self.merges],
            "special_tokens": self.special_tokens,
            "device": self.device,
        }, ensure_ascii=False, indent=2), encoding="utf-8")


def _pretokens(path: str | os.PathLike, specials: list[str],
               chunk_size: int) -> Iterator[bytes]:
    """Read training text in chunks and discard special-token boundaries."""
    with open(path, encoding="utf-8", newline="") as source:
        chunks = iter(lambda: source.read(chunk_size), "")
        for token in _stream_pretokens(chunks, specials):
            if isinstance(token, bytes):
                yield token


def _stream_pretokens(chunks: Iterable[str], specials: list[str]) -> Iterator[bytes | str]:
    """Yield ordinary byte pretokens and special strings from continuous text.

    Keep the earliest unresolved special prefix and the preceding final
    ordinary pretoken. Only this uncommitted tail is rescanned on the next read.
    Empty input chunks do not terminate the stream; exhaustion flushes it.
    """
    chunks = iter(chunks)
    special_re = (regex.compile("|".join(regex.escape(s) for s in
                  sorted(specials, key=lambda s: (-len(s), s)))) if specials else None)
    pending = ""
    while True:
        chunk = next(chunks, None)
        final = chunk is None
        if not final and not isinstance(chunk, str):
            raise TypeError("input chunks must be strings")
        if chunk == "":
            continue
        pending += chunk or ""
        unresolved_start = len(pending)
        if not final:
            for special in specials:
                for n in range(1, min(len(special), len(pending) + 1)):
                    if pending.endswith(special[:n]):
                        unresolved_start = min(unresolved_start, len(pending) - n)
        start = 0
        if special_re:
            for match in special_re.finditer(pending):
                if match.end() > unresolved_start:
                    # A full match can overlap a still-incomplete special.
                    # Retain it from its own start, never tokenize any part
                    # of it as ordinary text while its boundary is pending.
                    unresolved_start = min(unresolved_start, match.start())
                    break
                for token in _PRETOKEN.finditer(pending[start:match.start()]):
                    yield token.group().encode("utf-8")
                yield match.group()
                start = match.end()
        pending = pending[start:]
        unresolved_start -= start
        if final:
            for token in _PRETOKEN.finditer(pending):
                yield token.group().encode("utf-8")
            return
        # GPT-2 contraction alternatives can replace multiple matches:
        # "'r" currently matches apostrophe + letters, but adding "e" makes
        # one "'re" pretoken. Do not submit the apostrophe-containing match.
        # Inspect only the ordinary prefix before an unresolved special; a
        # completed special has already flushed its preceding ordinary text.
        ordinary_prefix = pending[:unresolved_start]
        for prefix in _CONTRACTION_PREFIXES:
            if ordinary_prefix.endswith(prefix):
                unresolved_start -= len(prefix)
                break
        # One-match lookbehind emits complete stable pretokens immediately
        # without accumulating a chunk-sized list of regex matches.
        last_pretoken = None
        for match in _PRETOKEN.finditer(pending[:unresolved_start]):
            if last_pretoken is not None:
                yield last_pretoken.group().encode("utf-8")
            last_pretoken = match
        # The final pretoken can change when a special prefix fails or
        # future input extends it (including whitespace lookahead).
        commit_boundary = last_pretoken.start() if last_pretoken is not None else 0
        pending = pending[commit_boundary:]


def _batches(tokens: Iterable[bytes], size: int = 4096):
    batch = []
    for token in tokens:
        batch.append(token)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _count_batch(batch: list[bytes]) -> Counter:
    return Counter(batch)


def _pair_counts(words: list[tuple[int, ...]], weights: list[int],
                 indices: Iterable[int], device: str, torch_module) -> Counter:
    counts = Counter()
    if device == "cpu":
        for i in indices:
            for pair in zip(words[i], words[i][1:]):
                counts[pair] += weights[i]
        return counts
    # Bounded batches avoid allocating the entire corpus on the GPU.
    pairs, frequencies = [], []
    def flush():
        if not pairs:
            return
        t = torch_module.tensor(pairs, dtype=torch_module.int64, device=device)
        unique, inverse = torch_module.unique(t, dim=0, return_inverse=True)
        sums = torch_module.zeros(len(unique), dtype=torch_module.int64, device=device)
        sums.scatter_add_(0, inverse, torch_module.tensor(
            frequencies, dtype=torch_module.int64, device=device))
        for pair, weight in zip(unique.cpu().tolist(), sums.cpu().tolist()):
            counts[tuple(pair)] += weight
        pairs.clear()
        frequencies.clear()
    for i in indices:
        for pair in zip(words[i], words[i][1:]):
            pairs.append(pair)
            frequencies.append(weights[i])
            if len(pairs) >= 262144:
                flush()
    flush()
    return counts


def train_bpe_result(input_path: str | os.PathLike, vocab_size: int,
                     special_tokens: list[str] | None = None, *,
                     parallel: bool = False, num_workers: int | None = None,
                     device: str = "auto", chunk_size: int = 1 << 20) -> BPETrainingResult:
    """Train with descending byte-lexicographic tie breaking.

    ``parallel`` controls CPU pretoken counting; CUDA accelerates weighted
    pair counting. Merges are sequential and update only affected words.
    Special IDs follow the 256 byte IDs, in the supplied order. Training may
    stop below vocab_size if no ordinary pair remains.
    """
    specials = list(dict.fromkeys(special_tokens or []))
    if any(not isinstance(s, str) or not s for s in specials):
        raise ValueError("special tokens must be nonempty strings")
    if vocab_size < 256 + len(specials):
        raise ValueError("vocab_size must include 256 bytes and all special tokens")
    if chunk_size < 1 or (num_workers is not None and num_workers < 1):
        raise ValueError("chunk_size and num_workers must be positive")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    torch_module = None
    if device != "cpu":
        try:
            import torch as torch_module
        except ImportError:
            pass
        available = torch_module is not None and torch_module.cuda.is_available()
        if device == "cuda" and not available:
            raise RuntimeError("CUDA was requested but a CUDA-enabled torch/GPU is unavailable")
        device = "cuda" if available else "cpu"
    token_counts = Counter()
    batches = _batches(_pretokens(input_path, specials, chunk_size))
    if parallel:
        # Keep outstanding jobs bounded (Executor.map eagerly consumes input
        # on older Python versions).
        with ProcessPoolExecutor(max_workers=num_workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            workers = num_workers or (os.cpu_count() or 1)
            while True:
                jobs = []
                for _ in range(workers * 2):
                    batch = next(batches, None)
                    if batch is None:
                        break
                    jobs.append(pool.submit(_count_batch, batch))
                if not jobs:
                    break
                for job in jobs:
                    token_counts.update(job.result())
    else:
        for batch in batches:
            token_counts.update(_count_batch(batch))
    vocab = {i: bytes([i]) for i in range(256)}
    special_table = {s: 256 + i for i, s in enumerate(specials)}
    vocab.update({i: s.encode("utf-8") for s, i in special_table.items()})
    words = [tuple(token) for token in token_counts]
    weights = list(token_counts.values())
    counts = _pair_counts(words, weights, range(len(words)), device, torch_module)
    locations = defaultdict(set)
    for i, word in enumerate(words):
        for pair in zip(word, word[1:]):
            locations[pair].add(i)
    merges = []
    candidates = _PairHeap(counts, vocab)
    while len(vocab) < vocab_size and counts:
        pair = candidates.pop()
        affected = list(locations[pair])
        old = _pair_counts(words, weights, affected, device, torch_module)
        new_id = len(vocab)
        merges.append((vocab[pair[0]], vocab[pair[1]]))
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        for i in affected:
            word = words[i]
            for old_pair in set(zip(word, word[1:])):
                locations[old_pair].discard(i)
                if not locations[old_pair]:
                    del locations[old_pair]
            merged = []
            j = 0
            while j < len(word):
                if j + 1 < len(word) and (word[j], word[j + 1]) == pair:
                    merged.append(new_id)
                    j += 2
                else:
                    merged.append(word[j])
                    j += 1
            words[i] = tuple(merged)
            for new_pair in zip(merged, merged[1:]):
                locations[new_pair].add(i)
        new = _pair_counts(words, weights, affected, device, torch_module)
        counts.subtract(old)
        counts.update(new)
        for changed in old.keys() | new.keys():
            if counts[changed] <= 0:
                del counts[changed]
            # The selected pair was popped and must be refreshed even if its
            # frequency is unchanged. Others only need updates on count changes.
            if changed == pair or old[changed] != new[changed]:
                candidates.refresh(changed)
        candidates.compact()
    return BPETrainingResult(vocab, merges, special_table, device)


def train_bpe(input_path: str | os.PathLike, vocab_size: int,
              special_tokens: list[str] | None = None, **kwargs
              ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Assignment-compatible return value: (vocab, ordered merges)."""
    result = train_bpe_result(input_path, vocab_size, special_tokens, **kwargs)
    return result.vocab, result.merges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--special-token", action="append", default=[])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--parallel", action="store_true")
    mode.add_argument("--serial", action="store_true")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="bpe_model.json")
    args = parser.parse_args()
    result = train_bpe_result(args.input_path, args.vocab_size, args.special_token,
                              parallel=args.parallel, num_workers=args.num_workers,
                              device=args.device)
    result.save(args.output)
    print(f"Saved {len(result.vocab)} tokens, {len(result.merges)} merges "
          f"to {args.output} (device={result.device})")


if __name__ == "__main__":
    main()
