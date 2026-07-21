"""Offline exporter contract: saved samples reproduce the committed diff."""
import os
import subprocess
import sys
import tempfile


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPORTER = os.path.join(ROOT, "eval", "export_run.py")
EXPECTED = os.path.join(ROOT, "results", "diffs", "diff-v1-158-vs-v2-158.md")


def test_compare_is_offline_and_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "diff.md")
        completed = subprocess.run(
            [sys.executable, "-S", EXPORTER, "--compare", "v1-158", "v2-158", "--out", out],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "offline" in completed.stdout
        with open(out, encoding="utf-8") as generated:
            actual = generated.read()
        with open(EXPECTED, encoding="utf-8") as committed:
            expected = committed.read()
        assert actual == expected
        assert "найдено gold-групп" in actual
        assert "найдено фактов" not in actual


if __name__ == "__main__":
    test_compare_is_offline_and_deterministic()
    print("ok test_compare_is_offline_and_deterministic")
    print("all export tests passed")
