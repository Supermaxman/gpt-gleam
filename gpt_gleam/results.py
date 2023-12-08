import os
from typing import Optional

import pandas as pd


class TabularResultsWriter:
    def __init__(self, file_path: str, point_metrics: Optional[list[str]] = None):
        self.file_path = file_path
        self.file = None
        self.point_metrics = point_metrics or ["F1", "P", "R"]

    def __enter__(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.file = open(self.file_path, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def _format_values(self, results: dict[str, float]):
        f_results = {}
        for k, v in results.items():
            k = k.replace("_", " ").title()
            if any(k.endswith(f" {m}") for m in self.point_metrics):
                v = f"{round(v * 100, 1):.1f}"
            else:
                v = f"{round(v, 3):.3f}"
            f_results[k] = v
        return f_results

    def write(self, results: dict[str, float]):
        results = self._format_values(results)
        df = pd.DataFrame.from_records([results])
        self.file.write(df.to_markdown(index=False))
        print(df.to_markdown(index=False))
        self.file.flush()
        os.fsync(self.file.fileno())
