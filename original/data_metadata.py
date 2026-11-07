"""Validate prepared datasets and fingerprint them before training."""
import hashlib
import json
import os
import tomllib
from pathlib import Path

import numpy as np

PREPROCESSING_VERSION = 1


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_metadata(directory):
    """Normalize Python JSON or Julia TOML preprocessing metadata."""
    directory = Path(directory)
    paths = [p for p in (directory / "metadata.json", directory / "metadata.toml") if p.exists()]
    if len(paths) != 1:
        raise ValueError(f"需要唯一的 metadata.json 或 metadata.toml: {directory}")
    path = paths[0]
    text = path.read_text(encoding="utf-8")
    data = tomllib.loads(text) if path.suffix == ".toml" else json.loads(text)
    if path.suffix == ".toml":
        version = data.get("format_version")
        if version not in (1, 2) or data.get("has_header") is not False:
            raise ValueError("不支持的 Julia metadata 格式或文件头")
        data = dict(data)
        data["preprocessing_version"] = f"julia:{version}"
        data["dtype"] = data["numpy_dtype"]
        data["tokenizer_path"] = data["model_path"]
        data["tokenizer_id"] = "sha256:" + data["model_sha256"]
        data["data_sha256"] = {}
        for role, source in [("train", "train"), ("val", "validation")]:
            entry = data[source]
            data[f"{role}_path"] = str((directory / entry["file"]).resolve())
            data[f"{role}_tokens"] = entry["tokens"]
            data["data_sha256"][role] = entry["sha256"]
        sources = data.get("sources", [])
        data["sources"] = {role: sources[i].get("path") if len(sources) > i else None
                           for i, role in enumerate(("train", "val"))}
        data["split_validation"] = data.get("automatic_split_enabled", False)
    return path, data


def validate_data_metadata(config):
    """Read adjacent metadata.json files and verify actual binary contents."""
    train_path, val_path = Path(config.train_path).resolve(), Path(config.val_path).resolve()
    if os.path.samefile(train_path, val_path):
        raise ValueError("训练集与验证集指向同一数据源")
    dtype = np.dtype(config.token_dtype)
    if not np.issubdtype(dtype, np.integer):
        raise ValueError("token dtype 必须为整数类型")
    fingerprints = {}
    for role, path in [("train", train_path), ("val", val_path)]:
        metadata_path, metadata = read_metadata(path.parent)
        if metadata.get("preprocessing_version") not in (PREPROCESSING_VERSION, "julia:1", "julia:2"):
            raise ValueError(f"{role} preprocessing_version 缺失或不匹配")
        tokenizer_id = metadata.get("tokenizer_id")
        if not isinstance(tokenizer_id, str) or not tokenizer_id.startswith("sha256:"):
            raise ValueError(f"{role} tokenizer_id 缺失或无效")
        tokenizer_path = Path(metadata.get("tokenizer_path", ""))
        if not tokenizer_path.is_absolute() and not tokenizer_path.is_file():
            tokenizer_path = metadata_path.parent / tokenizer_path
        if not tokenizer_path.is_file() or "sha256:" + file_sha256(tokenizer_path) != tokenizer_id:
            raise ValueError(f"{role} tokenizer 标识与模型文件不匹配: {tokenizer_path}")
        from bpe import BPE
        tokenizer = BPE.from_file(tokenizer_path)
        if max(tokenizer.vocab) + 1 != metadata.get("vocab_size"):
            raise ValueError(f"{role} tokenizer vocab_size 与 metadata 不匹配")
        if "special_tokens" in metadata and metadata["special_tokens"] != tokenizer.special_tokens:
            raise ValueError(f"{role} tokenizer special_tokens 与 metadata 不匹配")
        if metadata.get("vocab_size") != config.vocab_size:
            raise ValueError(f"{role} vocab_size 不匹配: metadata={metadata.get('vocab_size')}, config={config.vocab_size}")
        if np.dtype(metadata["dtype"]) != dtype:
            raise ValueError(f"{role} dtype 不匹配: metadata={metadata['dtype']}, config={config.token_dtype}")
        declared = metadata.get(f"{role}_path")
        if not declared:
            raise ValueError(f"metadata 未声明 {role}_path")
        declared_path = Path(declared)
        candidates = [declared_path.resolve(), (metadata_path.parent / declared_path).resolve()]
        if path not in candidates:
            raise ValueError(f"{role}_path 与 metadata 不匹配: {path}")
        count = metadata.get(f"{role}_tokens")
        if not isinstance(count, int) or count < config.sequence_length + 1:
            raise ValueError(f"{role}_tokens 不足以采样或无效: {count}")
        size = path.stat().st_size
        if size != count * dtype.itemsize:
            raise ValueError(f"{role} 二进制文件长度不匹配: 实际 {size} 字节, 预期 {count * dtype.itemsize}")
        tokens = np.memmap(path, dtype=dtype, mode="r")
        try:
            for start in range(0, count, 1 << 20):
                batch = tokens[start:start + (1 << 20)]
                if np.any((batch < 0) | (batch >= config.vocab_size)):
                    raise ValueError(f"{role} 二进制数据包含越界 token ID")
        finally:
            del tokens
        actual_hash = file_sha256(path)
        if metadata.get("data_sha256", {}).get(role) != actual_hash:
            raise ValueError(f"{role} 数据文件 SHA-256 与 metadata 不匹配")
        fingerprints[role] = {"path": str(path), "bytes": size, "tokens": count,
                              "dtype": dtype.str, "vocab_size": config.vocab_size,
                              "sha256": actual_hash, "tokenizer_id": tokenizer_id,
                              "preprocessing_version": metadata["preprocessing_version"],
                              "source": metadata.get("sources", {}).get(role),
                              "split_validation": metadata.get("split_validation", False)}
    if fingerprints["train"]["sha256"] == fingerprints["val"]["sha256"]:
        raise ValueError("训练集与验证集内容完全相同，可能来自同一数据源")
    a, b = fingerprints["train"], fingerprints["val"]
    if a["tokenizer_id"] != b["tokenizer_id"]:
        raise ValueError("训练集与验证集的 tokenizer 标识不一致")
    if a["source"] and b["source"] and a["source"] == b["source"]:
        if not (a["split_validation"] and b["split_validation"]):
            raise ValueError("训练集与验证集来自同一源文件且未执行文档划分")
    return fingerprints


def validate_resume_data(extra, current):
    saved = extra.get("data_metadata")
    if saved is None:
        raise ValueError("检查点缺少 data_metadata，无法核对恢复数据")
    if saved != current:
        differences = []
        for role in current:
            for name, value in current[role].items():
                previous = saved.get(role, {}).get(name)
                if previous != value:
                    differences.append(f"{role}.{name}: checkpoint={previous!r}, current={value!r}")
        raise ValueError("恢复数据与检查点不匹配:\n" + "\n".join(differences))
