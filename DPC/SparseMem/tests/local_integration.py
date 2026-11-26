"""Offline end-to-end test with local model files and synthetic HTTP completions."""

import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparsemem.run import mock


class FixtureEndpoint(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if payload["model"].startswith("gpt-4o-mini"):
            text, usage = "yes", {"prompt_tokens": 100, "completion_tokens": 1}
        else:
            stage = "construct" if "response_format" in payload else "query"
            text, usage = mock(payload["messages"], payload["max_tokens"], stage)
        result = dict(
            id="fixture",
            object="chat.completion",
            created=0,
            model=payload["model"],
            usage=dict(usage, total_tokens=sum(usage.values())),
            choices=[
                dict(
                    index=0,
                    finish_reason="stop",
                    message=dict(role="assistant", content=text),
                )
            ],
        )
        body = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureEndpoint)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    env = dict(
        os.environ,
        DPC_LLM_BASE_URL=url,
        DPC_LLM_API_KEY="test",
        OPENAI_BASE_URL=url,
        OPENAI_API_KEY="test",
        HF_HUB_OFFLINE="1",
        USE_TF="0",
        USE_FLAX="0",
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"
            data = Path(__file__).with_name("fixture.json")
            cmd = [
                sys.executable,
                "-m",
                "sparsemem.run",
                "--data",
                str(data),
                "--out",
                str(root),
                "--mode",
                "full_extract",
                "--execute",
            ]
            subprocess.run(cmd, env=env, check=True)
            subprocess.run(cmd + ["--resume"], env=env, check=True)
            metrics = json.loads((root / "metrics.json").read_text())
            assert (
                metrics["answer_failures"] == 0
                and metrics["construct_calls"] > 0
                and metrics["query_calls"] == 1
                and metrics["current_invocation"]["total_generation_tokens"] == 0
            )
            judge = [
                sys.executable,
                "-m",
                "sparsemem.evaluate",
                "--data",
                str(data),
                "--run",
                str(root),
                "--execute",
            ]
            subprocess.run(judge, env=env, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(judge, env=env, check=True, stdout=subprocess.DEVNULL)
            result = json.loads((root / "judge/gpt-4o-mini/metrics.json").read_text())
            assert result["judge_calls"] == 1 and result["known_output"] == 1
            print(
                "PASS: local MiniLM, Qwen tokenizer, P1/P3 HTTP, official judge, resume. Synthetic responses only."
            )
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
