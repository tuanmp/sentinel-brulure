"""Run the HLS evaluation under MLflow tracking and log metrics."""
import subprocess
import sys
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent.parent


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    mlflow.set_experiment("burn-scar-benchmark")
    with mlflow.start_run(run_name=f"hls-eval-{limit or 'full'}") as run:
        cmd = [sys.executable, str(ROOT / "scripts" / "run_evaluation.py")]
        if limit:
            cmd += ["--limit", str(limit)]
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        artifact_dir = max(
            (ROOT / "reports" / "evaluation").glob("*/"),
            key=lambda p: p.stat().st_mtime,
        )
        import json

        summary = json.loads(
            (artifact_dir / "summary.json").read_text(encoding="utf-8")
        )
        for group, metrics in summary.items():
            if isinstance(metrics, dict):
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"{group}.{name}", value)
        mlflow.log_artifacts(str(artifact_dir))
        print(f"Run logged: {run.info.run_id}")


if __name__ == "__main__":
    main()
