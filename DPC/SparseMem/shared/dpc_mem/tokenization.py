import os
import re
from pathlib import Path
from .io import digest, file_hash


class FixtureTokenizer:
    """Offline tests only. Never reported as real model tokens."""
    exact = False
    identity = "fixture-regex-NOT-MODEL-TOKENS"

    def count(self, text):
        return len(re.findall(r"[\u4e00-\u9fff]|\w+|[^\w\s]", text))

    def messages(self, messages):
        return 8 + sum(4 + self.count(m["content"]) for m in messages)


class HFTokenizer:
    exact = True

    def __init__(self, model_path):
        from transformers import AutoTokenizer
        p = Path(os.path.expandvars(model_path)).expanduser()
        if not p.is_dir():
            raise ValueError("Set DPC_TOKENIZER_PATH to an existing reader tokenizer directory")
        self.tokenizer = AutoTokenizer.from_pretrained(str(p), local_files_only=True,
                                                      trust_remote_code=False)
        files = {q.name: file_hash(q) for q in p.iterdir()
                 if q.is_file() and ("token" in q.name or q.name in {"vocab.json", "merges.txt", "chat_template.jinja"})}
        if not files:
            raise ValueError("Cannot fingerprint tokenizer")
        self.identity = digest(files)

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def messages(self, messages):
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True,
                   add_generation_prompt=True, enable_thinking=False))


def make_tokenizer(config, fixture=False):
    return FixtureTokenizer() if fixture else HFTokenizer(config["tokenizer_path"])
