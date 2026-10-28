"""Dispatch individual rollout cases to persistent, independent GPU workers."""

import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import urllib.parse
import urllib.request

from .runtime import atomic_json_save


def request(url, path, worker):
    with urllib.request.urlopen(f"{url}/{path}?{urllib.parse.urlencode({'worker': worker})}", timeout=30) as response:
        return json.load(response)


def evaluation_cases(environments, args):
    if not args.dispatch_url:
        for env in environments:
            yield env, None
        return
    worker = str(Path(args.output_dir).resolve())
    while True:
        case = request(args.dispatch_url, "claim", worker)
        if case is None:
            return
        yield case["env_id"], case["episode"]
        request(args.dispatch_url, "complete", worker)


class RolloutQueue:
    def __init__(self, root, environments, episodes, seed, max_steps, workers):
        self.root, self.workers = Path(root), workers
        self.seed, self.max_steps = seed, max_steps
        self.pending, self.running, self.finished = deque(), {}, set()
        self.completed = 0
        self.total = len(environments) * episodes
        self.lock = threading.Lock()
        for env in environments:
            for episode in range(episodes):
                case = dict(env_id=env, episode=episode)
                if self.metrics_path(case).exists():
                    self.verify(case)
                    self.completed += 1
                else:
                    self.pending.append(case)
        self.save()

    def metrics_path(self, case):
        return self.root / case["env_id"] / f"episode_{case['episode']:03d}" / "metrics.json"

    def verify(self, case):
        row = json.loads(self.metrics_path(case).read_text())
        assert row["env_id"] == case["env_id"]
        assert row["seed"] == self.seed + case["episode"]
        assert row["max_episode_steps"] == self.max_steps

    def save(self):
        atomic_json_save(dict(total=self.total, completed=self.completed, pending=len(self.pending),
                             running=self.running, finished_workers=len(self.finished)), self.root / "dispatch_status.json")

    def claim(self, worker):
        with self.lock:
            assert worker not in self.running
            if self.pending:
                case = self.pending.popleft()
                self.running[worker] = case
            else:
                case = None
                self.finished.add(worker)
            self.save()
            return case

    def complete(self, worker):
        with self.lock:
            case = self.running[worker]
            self.verify(case)
            del self.running[worker]
            self.completed += 1
            self.save()
            print(json.dumps(dict(event="rollout_completed", worker=worker, case=case,
                                  completed=self.completed, total=self.total)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--icl-guidance-scale", type=float, default=5)
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    environments = Path(args.eval_config).read_text().split()
    protocol = dict(checkpoint=str(checkpoint), checkpoint_mtime_ns=checkpoint.stat().st_mtime_ns,
                    episodes=args.episodes, seed=args.seed, max_steps=args.max_steps,
                    icl_guidance_scale=args.icl_guidance_scale, environments=environments)
    target = root / "protocol.json"
    if target.exists():
        assert json.loads(target.read_text()) == protocol
    atomic_json_save(protocol, target)
    queue = RolloutQueue(root, environments, args.episodes, args.seed, args.max_steps, args.workers)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            worker = urllib.parse.parse_qs(parsed.query)["worker"][0]
            if parsed.path == "/claim":
                result = queue.claim(worker)
            elif parsed.path == "/complete":
                queue.complete(worker)
                result = {"ok": True}
            else:
                self.send_error(404)
                return
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            if len(queue.finished) == queue.workers:
                threading.Thread(target=self.server.shutdown, daemon=True).start()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    atomic_json_save(dict(port=args.port, workers=args.workers), root / "ready.json")
    print(f"Dynamic rollout dispatcher ready on port{args.port}", flush=True)
    server.serve_forever()
    server.server_close()
    assert queue.completed == queue.total


if __name__ == "__main__":
    main()
