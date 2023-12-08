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
        formats = []
        for k, v in results.items():
            k = k.replace("_", " ").title()
            if any(k.endswith(f" {m}") for m in self.point_metrics):
                v = f"{v * 100:.1f}"
                formats.append(".1f")
            else:
                v = f"{v:.3f}"
                formats.append(".3f")
            f_results[k] = v
        return f_results, formats

    def write(self, results: dict[str, float]):
        results, formats = self._format_values(results)
        print(results)
        df = pd.DataFrame.from_records([results], coerce_float=False)
        print(df)
        self.file.write(df.to_markdown(index=False, floatfmt=formats))
        print(df.to_markdown(index=False, floatfmt=".1f"))
        self.file.flush()
        os.fsync(self.file.fileno())
