from __future__ import annotations

import os
import tempfile
import unittest

from modules.files_renamer import (
    BatchRenameConfig,
    BatchRenameEngine,
    Config,
    MatchMode,
    RenameRule,
    RenameStatus,
)


class BatchRenameEngineTests(unittest.TestCase):
    def test_prefix_mode_renames_matching_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._touch(tmp_dir, "-demo.txt")
            self._touch(tmp_dir, "keep.txt")

            summary = self._run_engine(
                tmp_dir,
                RenameRule(
                    mode=MatchMode.PREFIX,
                    pattern="-",
                    replace_with="",
                    include_extension=False,
                    case_sensitive=False,
                ),
            )

            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "demo.txt")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "keep.txt")))
            self.assertEqual(summary.stats[RenameStatus.SUCCESS], 1)
            self.assertEqual(summary.stats[RenameStatus.SKIPPED_NO_MATCH], 1)

    def test_all_mode_replaces_all_occurrences(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._touch(tmp_dir, "a a a.txt")

            summary = self._run_engine(
                tmp_dir,
                RenameRule(
                    mode=MatchMode.ALL,
                    pattern=" ",
                    replace_with="_",
                    include_extension=False,
                    case_sensitive=False,
                ),
            )

            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "a_a_a.txt")))
            self.assertEqual(summary.stats[RenameStatus.SUCCESS], 1)

    def test_body_mode_preserves_prefix_and_replaces_later_matches(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._touch(tmp_dir, "-a-b.txt")

            summary = self._run_engine(
                tmp_dir,
                RenameRule(
                    mode=MatchMode.BODY,
                    pattern="-",
                    replace_with="_",
                    include_extension=False,
                    case_sensitive=False,
                ),
            )

            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "-a_b.txt")))
            self.assertEqual(summary.stats[RenameStatus.SUCCESS], 1)

    def test_between_mode_replaces_text_between_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._touch(tmp_dir, "数学二应用第01讲.txt")

            summary = self._run_engine(
                tmp_dir,
                RenameRule(
                    mode=MatchMode.BETWEEN,
                    replace_with="",
                    include_extension=False,
                    case_sensitive=False,
                    start_pattern="数学二",
                    end_pattern="第",
                ),
            )

            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "数学二第01讲.txt")))
            self.assertEqual(summary.stats[RenameStatus.SUCCESS], 1)

    def test_recursive_scan_ignores_hidden_entries_and_detects_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.mkdir(os.path.join(tmp_dir, "visible"))
            os.mkdir(os.path.join(tmp_dir, ".hidden"))
            self._touch(tmp_dir, "a.txt")
            self._touch(tmp_dir, "-a.txt")
            self._touch(os.path.join(tmp_dir, "visible"), "-nested.txt")
            self._touch(os.path.join(tmp_dir, ".hidden"), "-secret.txt")
            self._touch(tmp_dir, ".gitignore")

            summary = self._run_engine(
                tmp_dir,
                RenameRule(
                    mode=MatchMode.PREFIX,
                    pattern="-",
                    replace_with="",
                    include_extension=False,
                    case_sensitive=False,
                ),
                recursive=True,
            )

            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "a.txt")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "-a.txt")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "visible", "nested.txt")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, ".hidden", "-secret.txt")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, ".gitignore")))
            self.assertEqual(summary.file_count, 3)
            self.assertEqual(summary.stats[RenameStatus.SUCCESS], 1)
            self.assertEqual(summary.stats[RenameStatus.SKIPPED_CONFLICT], 1)

    def _run_engine(
        self,
        target_dir: str,
        rule: RenameRule,
        recursive: bool = False,
    ):
        engine = BatchRenameEngine(
            BatchRenameConfig(
                target_dir=target_dir,
                rule=rule,
                recursive=recursive,
                max_workers=Config.DEFAULT_WORKERS,
            )
        )
        return engine.run()

    @staticmethod
    def _touch(directory: str, file_name: str):
        with open(os.path.join(directory, file_name), "w", encoding="utf-8") as fh:
            fh.write("demo")


if __name__ == "__main__":
    unittest.main()
