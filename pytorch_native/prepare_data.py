"""Stream text through BPE into raw binary train/validation token files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from bpe import BPE
from data_metadata import PREPROCESSING_VERSION, file_sha256


def _documents(path: str | os.PathLike, separator: str,
               chunk_size: int):
    """Yield document chunks; None markers terminate documents.

    Retain the special separator at the end of each document. Newlines do not
    define document boundaries.
    """
    with open(path, encoding="utf-8", newline="") as source:
        pending = ""
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                if pending:
                    yield pending
                yield None
                return
            pending += chunk
            while True:
                index = pending.find(separator)
                if index < 0:
                    break
                end = index + len(separator)
                yield pending[:end]
                yield None
                pending = pending[end:]
            safe_end = max(0, len(pending) - len(separator) + 1)
            if safe_end:
                yield pending[:safe_end]
                pending = pending[safe_end:]


class _TokenWriter:
    def __init__(self, stream, dtype, buffer_size, vocab_size):
        self.stream = stream
        self.dtype = dtype
        self.buffer_size = buffer_size
        self.buffer = []
        self.count = 0
        self.vocab_size = vocab_size
        self.digest = hashlib.sha256()

    def write(self, ids):
        for token_id in ids:
            self.buffer.append(token_id)
            if len(self.buffer) >= self.buffer_size:
                self.flush()

    def flush(self):
        if self.buffer:
            # Validate before narrowing to the binary storage dtype.
            if any(not isinstance(i, (int, np.integer)) or not 0 <= i < self.vocab_size
                   for i in self.buffer):
                raise ValueError("token ID must be an integer in [0, vocab_size)")
            array = np.asarray(self.buffer, dtype=self.dtype)
            payload = array.tobytes()
            self.stream.write(payload)
            self.digest.update(payload)
            self.count += len(self.buffer)
            self.buffer.clear()


def prepare_data(input_path: str | os.PathLike, tokenizer_path: str | os.PathLike,
                 output_dir: str | os.PathLike = "data", *,
                 validation_path: str | os.PathLike | None = None,
                 split_validation: bool = False,
                 validation_ratio: float = 0.1, seed: int = 42,
                 document_separator: str | None = None, dtype: str = "int64",
                 chunk_size: int = 1 << 20, buffer_size: int = 65536) -> dict:
    """Write train.bin/valid.bin plus metadata.json without loading the corpus.

    If validation_path is provided, both files are encoded continuously and
    retain their existing split. Automatic splitting defaults off: without a
    validation file, only train.bin is produced. Enable split_validation and
    supply a special document_separator to assign whole documents randomly
    with probability validation_ratio. Newlines never split documents.
    """
    if split_validation and validation_path is not None:
        raise ValueError("split_validation cannot be combined with validation_path")
    if split_validation and document_separator is None:
        raise ValueError("split_validation requires a special document_separator")
    if split_validation and not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must lie strictly between 0 and 1")
    if chunk_size <= 0 or buffer_size <= 0:
        raise ValueError("chunk_size and buffer_size must be positive")
    storage_dtype = np.dtype(dtype)
    if not np.issubdtype(storage_dtype, np.integer):
        raise ValueError("dtype must be an integer dtype")
    tokenizer = BPE.from_file(tokenizer_path)
    tokenizer_id = "sha256:" + file_sha256(tokenizer_path)
    if document_separator is not None and document_separator not in tokenizer.special_tokens:
        raise ValueError("document_separator must be a tokenizer special token")
    bounds = np.iinfo(storage_dtype)
    if min(tokenizer.vocab) < 0:
        raise ValueError("tokenizer IDs must be nonnegative")
    if min(tokenizer.vocab) < bounds.min or max(tokenizer.vocab) > bounds.max:
        raise ValueError("dtype cannot represent all tokenizer IDs")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_path, val_path = output / "train.bin", output / "valid.bin"
    inputs = [Path(input_path), Path(tokenizer_path)]
    if validation_path is not None:
        inputs.append(Path(validation_path))
    if any(destination.resolve() == source.resolve()
           for destination in (train_path, val_path, output / "metadata.json") for source in inputs):
        raise ValueError("output paths must not overwrite inputs")
    temporary = []
    try:
        with tempfile.NamedTemporaryFile(dir=output, delete=False) as train_stream, \
                tempfile.NamedTemporaryFile(dir=output, delete=False) as val_stream:
            temporary = [Path(train_stream.name), Path(val_stream.name)]
            vocab_size = max(tokenizer.vocab) + 1
            train_writer = _TokenWriter(train_stream, storage_dtype, buffer_size, vocab_size)
            val_writer = _TokenWriter(val_stream, storage_dtype, buffer_size, vocab_size)
            if validation_path is not None:
                for source_path, writer in [(input_path, train_writer), (validation_path, val_writer)]:
                    with open(source_path, encoding="utf-8", newline="") as source:
                        chunks = iter(lambda: source.read(chunk_size), "")
                        writer.write(tokenizer.encode_iterable(chunks))
            elif split_validation:
                rng = np.random.default_rng(seed)
                documents = iter(_documents(input_path, document_separator, chunk_size))
                while True:
                    first = next(documents, None)
                    if first is None:
                        break
                    # Consume a single document lazily through the encoder.
                    def chunks():
                        yield first
                        for chunk in documents:
                            if chunk is None:
                                return
                            yield chunk
                    writer = val_writer if rng.random() < validation_ratio else train_writer
                    writer.write(tokenizer.encode_iterable(chunks()))
            else:
                with open(input_path, encoding="utf-8", newline="") as source:
                    chunks = iter(lambda: source.read(chunk_size), "")
                    train_writer.write(tokenizer.encode_iterable(chunks))
            train_writer.flush()
            val_writer.flush()
            has_validation = split_validation or validation_path is not None
            if not train_writer.count or (has_validation and not val_writer.count):
                raise ValueError("split produced an empty dataset; provide more documents or independent validation text")
            for stream in (train_stream, val_stream):
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(temporary[0], train_path)
        if has_validation:
            os.replace(temporary[1], val_path)
        metadata = {"train_path": str(train_path), "val_path": str(val_path) if has_validation else None,
                    "train_tokens": train_writer.count, "val_tokens": val_writer.count,
                    "dtype": storage_dtype.str, "vocab_size": max(tokenizer.vocab) + 1,
                    "seed": seed, "split_validation": split_validation,
                    "validation_ratio": validation_ratio if split_validation else None,
                    "document_separator": document_separator,
                    "tokenizer_path": str(Path(tokenizer_path).resolve()),
                    "tokenizer_id": tokenizer_id,
                    "preprocessing_version": PREPROCESSING_VERSION,
                    "data_sha256": {"train": train_writer.digest.hexdigest(),
                                    "val": val_writer.digest.hexdigest() if has_validation else None}}
        metadata["sources"] = {"train": str(Path(input_path).resolve()),
                               "val": str(Path(validation_path).resolve()) if validation_path is not None
                               else str(Path(input_path).resolve()) if split_validation else None}
        (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
        return metadata
    finally:
        for path in temporary:
            if path.exists():
                path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--validation-path")
    parser.add_argument("--split-validation", action="store_true",
                        help="split whole documents using --document-separator (default off)")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--document-separator")
    parser.add_argument("--dtype", default="int64")
    parser.add_argument("--chunk-size", type=int, default=1 << 20)
    parser.add_argument("--buffer-size", type=int, default=65536)
    metadata = prepare_data(**vars(parser.parse_args()))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
