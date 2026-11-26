from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()


def json_object(value):
    value = value.strip()
    if value.startswith("```json") and value.endswith("```"):
        value = value[7:-3].strip()
    elif value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
    result = json.loads(value)
    if not isinstance(result, dict):
        raise ValueError("Expected one JSON object")
    return result


def key(value):
    return digest(value)[:24]


def code_hash():
    root = Path(__file__).parent
    return digest({p.name: file_hash(p) for p in sorted(root.glob("*.py"))})


class RunLock:
    """Kernel-owned lock: automatically released when a process exits or crashes."""
    def __init__(self, root):
        self.path = Path(root) / ".writer.lock"
        self.handle = None

    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            raise RuntimeError("Another writer currently owns this run") from None
        self.handle.seek(0);self.handle.truncate();self.handle.write(str(os.getpid()));self.handle.flush()
        return self

    def __exit__(self, *args):
        import fcntl
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        # Do not unlink: a waiting process may already hold this inode.
