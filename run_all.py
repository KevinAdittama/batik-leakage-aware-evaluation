"""Jalankan pipeline penelitian sampai robustness checks untuk submission."""

import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

from pipeline_config import DEEP_LEARNING_DIR, EXTERNAL_RESULT_DIR, LOG_DIR, RESULTS_DIR


HERE = Path(__file__).resolve().parent
STAGES = [
    ("Audit dataset", "01_audit_dataset.py"),
    ("Preview preprocessing original", "02_preview_preprocessing.py"),
    ("Augmentasi seimbang", "03_augment_dataset.py"),
    ("Preview augmentasi", "04_preview_augmented.py"),
    ("Ekstraksi enam fitur", "05_extract_features.py"),
    ("Analisis fitur asli", "06_feature_analysis.py"),
    ("5-fold CV dan ablasi fitur", "07_eval_5fold.py"),
    ("Final train dan uji eksternal", "08_train_external.py"),
    ("Tabel paper", "09_tabel_paper.py"),
    ("Analisis error", "10_analisis_error.py"),
    ("Baseline deep learning", "11_eval_deep_learning.py"),
    ("Robustness checks submission", "12_submission_robustness.py"),
]


class TeeLogger:
    """Tulis output pipeline ke terminal dan file log."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("w", encoding="utf-8", newline="")

    def write(self, text: str) -> None:
        console_encoding = sys.stdout.encoding or "utf-8"
        console_text = text.encode(console_encoding, errors="replace").decode(
            console_encoding, errors="replace"
        )
        print(console_text, end="", flush=True)
        self.handle.write(text)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def run_and_log(command: list[str], cwd: Path, logger: TeeLogger) -> int:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.write(line)
    return process.wait()


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"run_all_{_dt.datetime.now():%Y%m%d_%H%M%S}.log"
    logger = TeeLogger(log_path)
    try:
        logger.write(f"Log pipeline: {log_path}\n")
        for index, (title, script) in enumerate(STAGES, 1):
            logger.write("\n" + "#" * 78 + "\n")
            logger.write(f"[{index}/{len(STAGES)}] {title} — {script}\n")
            logger.write("#" * 78 + "\n")
            path = HERE / script
            if not path.exists():
                raise FileNotFoundError(f"Skrip wajib tidak ada: {path}")
            returncode = run_and_log([sys.executable, str(path)], HERE, logger)
            if returncode != 0:
                raise SystemExit(f"[GAGAL] {script}: exit code {returncode}")

        result_path = EXTERNAL_RESULT_DIR / "selected_model_result.json"
        selected = json.loads(result_path.read_text(encoding="utf-8"))
        logger.write("\n" + "=" * 78 + "\n")
        logger.write("PIPELINE SELESAI\n")
        logger.write("=" * 78 + "\n")
        logger.write(f"Model formal           : {selected['model']}\n")
        logger.write(
            f"CV macro-F1           : {selected['cv_f1_macro_mean']:.3f} "
            f"± {selected['cv_f1_macro_std']:.3f}\n"
        )
        logger.write(f"External macro-F1     : {selected['external_f1_macro']:.3f}\n")
        logger.write(f"Recall batik eksternal: {selected['external_recall_batik']:.3f}\n")
        logger.write(f"Recall non-batik      : {selected['external_recall_non_batik']:.3f}\n")

        dl_summary = DEEP_LEARNING_DIR / "dl_cv_summary.csv"
        if dl_summary.exists():
            logger.write(f"Baseline DL summary   : {dl_summary}\n")
        robustness_log = RESULTS_DIR / "12_submission_robustness" / "run.log"
        if robustness_log.exists():
            logger.write(f"Robustness diagnostics: {robustness_log}\n")
        logger.write(f"Seluruh hasil paper   : {RESULTS_DIR}\n")
        logger.write(f"Log lengkap pipeline  : {log_path}\n")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
