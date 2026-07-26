import subprocess
import sys


def test_bootstrap():

    result = subprocess.run(
        [sys.executable, "run.py"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "FedAegis-X" in result.stdout
