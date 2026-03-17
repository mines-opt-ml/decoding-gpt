"""Download Pile-10k parquet from HuggingFace and convert to JSONL.

Skips the download when the output file already exists and its SHA-256
matches the expected hash stored in ``pile_10k_hash.txt`` (git-tracked).
"""

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

URL: str = (
	"https://huggingface.co/datasets/NeelNanda/pile-10k/resolve/main/"
	"data/train-00000-of-00001-4746b8785c874cc7.parquet"
)

# Hash file lives next to this module, tracked in git.
_HASH_FILE: Path = Path(__file__).parent / "pile_10k_hash.txt"

_HASH_CHUNK_SIZE: int = 1 << 16  # 64 KiB


def _read_expected_hash() -> str:
	"""Read the expected SHA-256 hash from the git-tracked hash file."""
	return _HASH_FILE.read_text().strip()


def _compute_file_sha256(path: Path) -> str:
	"""Compute SHA-256 hex digest of *path*, streaming to limit memory."""
	h: hashlib._Hash = hashlib.sha256()
	with open(path, "rb") as f:
		while True:
			chunk: bytes = f.read(_HASH_CHUNK_SIZE)
			if not chunk:
				break
			h.update(chunk)
	return h.hexdigest()


def _file_is_fresh(output: Path) -> bool:
	"""Return ``True`` if *output* exists and its hash matches the expected hash."""
	if not output.is_file():
		return False
	if not _HASH_FILE.is_file():
		# No hash file checked in yet — cannot verify, must re-download.
		return False
	expected: str = _read_expected_hash()
	actual: str = _compute_file_sha256(output)
	match: bool = actual == expected
	if not match:
		print(f"Hash mismatch: expected {expected[:16]}…, got {actual[:16]}…")
	return match


def _download_and_convert(output: Path) -> str:
	"""Download the parquet, convert to JSONL, return SHA-256 of the result."""
	with tempfile.TemporaryDirectory() as tmp_dir:
		parquet_path: Path = Path(tmp_dir) / "pile_10k.parquet"
		print(f"Downloading {URL} ...")
		urllib.request.urlretrieve(URL, parquet_path)
		print(f"Saved parquet to {parquet_path}")
		df: pd.DataFrame = pd.read_parquet(parquet_path)

	output.parent.mkdir(parents=True, exist_ok=True)
	with open(output, "w") as f:
		for _, row in df.iterrows():
			line: dict[str, object] = {"text": row["text"], "meta": row["meta"]}
			f.write(json.dumps(line, ensure_ascii=False) + "\n")

	print(f"Wrote {len(df)} lines to {output}")
	file_hash: str = _compute_file_sha256(output)
	return file_hash


def main() -> None:
	parser: argparse.ArgumentParser = argparse.ArgumentParser(
		description="Download Pile-10k parquet and convert to JSONL.",
	)
	parser.add_argument(
		"output",
		type=Path,
		help="Output path for the JSONL file",
	)
	parser.add_argument(
		"--force",
		action="store_true",
		help="Force re-download even if file exists with correct hash",
	)
	parser.add_argument(
		"--update-hash",
		action="store_true",
		help="After downloading, update the hash file (for maintainers)",
	)
	args: argparse.Namespace = parser.parse_args()
	output: Path = args.output

	if not args.force and _file_is_fresh(output):
		print(f"Skipping download: {output} exists and hash matches")
		return

	file_hash: str = _download_and_convert(output)
	print(f"SHA-256: {file_hash}")

	if args.update_hash:
		_HASH_FILE.write_text(file_hash + "\n")
		print(f"Updated hash file: {_HASH_FILE}")


if __name__ == "__main__":
	main()
