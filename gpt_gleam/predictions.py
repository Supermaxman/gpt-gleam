import os

import ujson as json

from gpt_gleam.data import read_jsonl


class JsonlPredictionsWriter:
    def __init__(self, file_path: str, buffer_size: int = 100):
        self.file_path = file_path
        self.seen_ids = set()
        self.file = None
        self.writes = 0
        self.buffer_size = buffer_size

    def __enter__(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        if os.path.exists(self.file_path):
            for pred in read_jsonl(self.file_path):
                self.seen_ids.add(pred["id"])
        self.file = open(self.file_path, "a")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def add(self, pred: dict):
        self.file.write(json.dumps(pred) + "\n")
        self.writes += 1
        self.seen_ids.add(pred["id"])
        if self.writes % self.buffer_size == 0:
            self.file.flush()
            os.fsync(self.file.fileno())

    def __len__(self):
        return len(self.seen_ids)

    def __contains__(self, item):
        return item in self.seen_ids
