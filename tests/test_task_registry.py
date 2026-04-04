from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest

from core.task_registry import get_task_spec, get_task_specs


class TaskRegistryTests(unittest.TestCase):
    def test_task_spec_keys_are_unique(self):
        specs = get_task_specs()

        self.assertEqual(len(specs), len({spec.key for spec in specs}))

    def test_get_task_spec_by_key(self):
        spec = get_task_spec("subtitle-generation")

        self.assertEqual(spec.title, "字幕生成")
        self.assertEqual(spec.module_path, "modules.gen_subtitles")
        self.assertEqual(spec.class_name, "GenSubtitles")

    def test_core_registry_import_does_not_pull_gui_modules(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json, sys; "
                    "import core.task_registry; "
                    "print(json.dumps({"
                    "'ui': 'ui' in sys.modules, "
                    "'qfluentwidgets': 'qfluentwidgets' in sys.modules"
                    "}))"
                ),
            ],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        )

        imported_state = json.loads(completed.stdout.strip())
        self.assertFalse(imported_state["ui"])
        self.assertFalse(imported_state["qfluentwidgets"])


if __name__ == "__main__":
    unittest.main()
