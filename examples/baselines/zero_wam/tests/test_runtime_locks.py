"""CPU checks for launcher-fenced local locks."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch





class RunLockTests(unittest.TestCase):
    def test_local_override_is_exclusive_and_separates_runs(self):
        from examples.baselines.zero_wam.zero_wam.runtime import training_rank_lock

        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"IG_TRAIN_LOCK_ROOT": temp}):
            run = Path(temp) / "run_a"
            lock = training_rank_lock(run, 6)
            self.assertNotEqual(Path(lock.lock_file).parent, run)
            self.assertNotEqual(lock.lock_file, training_rank_lock(Path(temp) / "run_b", 6).lock_file)
            self.assertNotEqual(lock.lock_file, training_rank_lock(run, 7).lock_file)
            with lock:
                child = subprocess.run([sys.executable, "-c",
                    "from filelock import FileLock,Timeout;import sys\n"
                    "try: FileLock(sys.argv[1],timeout=0).acquire()\n"
                    "except Timeout: sys.exit(23)", lock.lock_file], capture_output=True, timeout=15)
                self.assertEqual(child.returncode, 23, child.stderr)
            with training_rank_lock(run, 6):
                pass

    def test_default_remains_shared_run_lock(self):
        from examples.baselines.zero_wam.zero_wam.runtime import training_rank_lock

        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Path(training_rank_lock(temp, 1).lock_file), Path(temp).resolve() / ".rank1.lock")


if __name__ == "__main__":
    unittest.main()
