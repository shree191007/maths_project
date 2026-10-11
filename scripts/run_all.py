"""Usage:  python scripts/run_all.py [data|1|2|3|4|all]

Writes figures to reports/figures and tables to reports/tables.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from climatematrix import pipeline  # noqa: E402

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    runners = {"data": pipeline.run_data, "1": pipeline.run_part1, "2": pipeline.run_part2, "3": pipeline.run_part3, "4": pipeline.run_part4}
    for key in (runners if which == "all" else [which]):
        print(f"Part {key} ...", flush=True)
        runners[key]()
    print("done -> reports/figures, reports/tables")
