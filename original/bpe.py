"""GPT-2 byte-level BPE tokenizer for JSON models saved by bpe_train."""
from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import regex

from bpe_train import GPT2_PATTERN, _stream_pretokens

REPLACEMENT_CHARACTER = "\ufffd"
_REPLACEMENT_BYTES = REPLACEMENT_CHARACTER.encode("utf-8")
_PRETOKEN = regex.compile(GPT2_PATTERN)


class BPE:
    """Construct from training tables, or load them with ``BPE.from_file``.

    Special tokens in input are recognized automatically, longest first.
    Ordinary tokens cannot merge across pretoken or special-token boundaries.
    A special-token list reuses existing vocabulary entries and appends missing
    entries after the largest vocabulary ID, without changing the input mapping.
    """

    def __init__(self, vocab: Mapping[int, bytes],
                 merges: Iterable[tuple[bytes, bytes]],
                 special_tokens: Mapping[str, int] | list[str] | None = None):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        if any(not isinstance(i, int) or not isinstance(b, bytes)
               for i, b in self.vocab.items()):
            raise ValueError("vocab must map integer IDs to bytes")
        if isinstance(special_tokens, list):
            self.special_tokens = {}
            byte_to_id = {b: i for i, b in self.vocab.items()}
            next_id = max(self.vocab, default=-1) + 1
            for token in special_tokens:
                if not isinstance(token, str) or not token:
                    raise ValueError("special tokens must be non-empty strings")
                value = token.encode("utf-8")
                if value not in byte_to_id:
                    byte_to_id[value] = next_id
                    self.vocab[next_id] = value
                    next_id += 1
                self.special_tokens[token] = byte_to_id[value]
        else:
            self.special_tokens = dict(special_tokens or {})
        for token, token_id in self.special_tokens.items():
            if not token or self.vocab.get(token_id) != token.encode("utf-8"):
                raise ValueError("special tokens must match their vocabulary entries")
        self._byte_to_id = {b: i for i, b in self.vocab.items()}
        if any(bytes([i]) not in self._byte_to_id for i in range(256)):
            raise ValueError("vocab must contain all 256 ordinary byte tokens")
        self._ranks = {}
        for rank, pair in enumerate(self.merges):
            if len(pair) != 2 or any(part not in self._byte_to_id for part in pair):
                raise ValueError("merge components must exist in the ordinary vocabulary")
            if pair[0] + pair[1] not in self._byte_to_id:
                raise ValueError("merged bytes must exist in the ordinary vocabulary")
            self._ranks.setdefault(tuple(pair), rank)
        self._special_re = (regex.compile("|".join(
            regex.escape(token) for token in sorted(
                self.special_tokens, key=lambda token: (-len(token), token))))
            if self.special_tokens else None)

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> BPE:
        """Read bpe_train's hexadecimal byte vocabulary and ordered merges."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        model = tomllib.loads(text) if path.suffix.lower() == ".toml" else json.loads(text)
        if model.get("pretoken_pattern", GPT2_PATTERN) != GPT2_PATTERN:
            raise ValueError("tokenizer pretoken_pattern is incompatible with GPT-2")
        if model.get("format_version", 1) != 1:
            raise ValueError("unsupported tokenizer format_version")
        return cls(
            {int(i): bytes.fromhex(value) for i, value in model["vocab"].items()},
            [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in model["merges"]],
            model.get("special_tokens", {}),
        )

    def _encode_piece(self, piece: bytes) -> list[int]:
        parts = [bytes([b]) for b in piece]
        while len(parts) > 1:
            candidates = ((self._ranks[pair], pair)
                          for pair in zip(parts, parts[1:]) if pair in self._ranks)
            best = min(candidates, default=None)
            if best is None:
                break
            pair = best[1]
            merged = []
            i = 0
            while i < len(parts):
                if i + 1 < len(parts) and (parts[i], parts[i + 1]) == pair:
                    merged.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        return [self._byte_to_id[part] for part in parts]

    def encode(self, text: str) -> list[int]:
        """Encode text using GPT-2 pretokens and the learned merge ranks."""
        return list(self._encode_text(text))

    def _encode_text(self, text: str) -> Iterator[int]:
        def encode_ordinary(segment: str) -> Iterator[int]:
            for match in _PRETOKEN.finditer(segment):
                yield from self._encode_piece(match.group().encode("utf-8"))

        start = 0
        if self._special_re:
            for match in self._special_re.finditer(text):
                yield from encode_ordinary(text[start:match.start()])
                yield self.special_tokens[match.group()]
                start = match.end()
        yield from encode_ordinary(text[start:])

    def encode_iterable(self, chunks: Iterable[str]) -> Iterator[int]:
        """Lazily encode chunks as one continuous text stream.

        Equivalent to encode(''.join(chunks)), including specials spanning
        chunks, contractions, whitespace lookahead and ordinary BPE merges.
        Emit stable pretokens immediately and retain only the uncommitted
        tail. The final pretoken is flushed when the iterable is exhausted.
        Memory depends on chunk and unfinished pretoken/special size, and
        more input may be needed before a token becomes safe to emit.
        """
        for token in _stream_pretokens(chunks, list(self.special_tokens)):
            if isinstance(token, str):
                yield self.special_tokens[token]
            else:
                yield from self._encode_piece(token)

    def encode_file(self, path: str | os.PathLike, *,
                    encoding: str = "utf-8") -> Iterator[int]:
        """Read lines lazily and encode them as a continuous stream.

        Memory depends on the longest line/pretoken, rather than file size.
        The file closes on exhaustion or generator.close(); when stopping
        early, close the generator explicitly or use contextlib.closing.
        """
        with open(path, encoding=encoding, newline="") as source:
            yield from self.encode_iterable(source)

    def decode(self, token_ids: Iterable[int]) -> str:
        """Decode joined bytes as UTF-8; unknown IDs become U+FFFD (�).

        Joining before decoding preserves characters split across byte tokens.
        Invalid or incomplete UTF-8 is also decoded with replacement characters.
        """
        return b"".join(self.vocab.get(i, _REPLACEMENT_BYTES)
                        for i in token_ids).decode("utf-8", errors="replace")


BPETokenizer = BPE
