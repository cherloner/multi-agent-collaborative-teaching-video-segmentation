import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from process_control import controlled_processes, run_process


class ControlAbort(RuntimeError):
    pass


class ProcessControlTests(unittest.TestCase):
    def test_successful_process_preserves_completed_process_contract(self) -> None:
        result = run_process(
            [sys.executable, "-c", "print('ready')"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "ready")

    def test_timeout_terminates_process_promptly(self) -> None:
        started = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            run_process(
                [sys.executable, "-c", "import time; time.sleep(5)"],
                timeout=0.1,
            )
        self.assertLess(time.monotonic() - started, 2.0)

    def test_control_abort_terminates_descendant_process_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            marker = Path(temp_dir) / "descendant-survived.txt"
            child_code = (
                "import pathlib,time;"
                "time.sleep(0.7);"
                f"pathlib.Path({str(marker)!r}).write_text('leak')"
            )
            parent_code = (
                "import subprocess,sys,time;"
                f"subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
                "time.sleep(5)"
            )
            abort_at = time.monotonic() + 0.1

            def check() -> None:
                if time.monotonic() >= abort_at:
                    raise ControlAbort("cancelled")

            started = time.monotonic()
            with self.assertRaises(ControlAbort):
                with controlled_processes(check):
                    run_process(
                        [sys.executable, "-c", parent_code],
                        timeout=10,
                    )
            self.assertLess(time.monotonic() - started, 2.0)
            time.sleep(0.9)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
