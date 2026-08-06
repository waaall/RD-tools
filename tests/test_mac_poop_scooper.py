from __future__ import annotations

import os
import tempfile
import unittest

from modules.mac_poop_scooper import MacPoopScooper


class MacPoopScooperTests(unittest.TestCase):
    def test_recursively_deletes_ds_store_and_preserves_appledouble_safety_rule(self):
        with tempfile.TemporaryDirectory() as work_dir:
            selected_dir = os.path.join(work_dir, "selected")
            nested_dir = os.path.join(selected_dir, "nested", "deep")
            os.makedirs(nested_dir)

            root_ds_store = os.path.join(selected_dir, ".DS_Store")
            nested_ds_store = os.path.join(nested_dir, ".DS_Store")
            similarly_named_file = os.path.join(nested_dir, ".DS_Store.backup")
            normal_file = os.path.join(nested_dir, "document.txt")
            paired_appledouble = os.path.join(nested_dir, "._document.txt")
            orphan_appledouble = os.path.join(nested_dir, "._orphan.txt")

            for path in (
                root_ds_store,
                nested_ds_store,
                similarly_named_file,
                normal_file,
                paired_appledouble,
                orphan_appledouble,
            ):
                self._touch(path)

            cleaner = MacPoopScooper(parallel=False)
            cleaner.set_work_folder(work_dir)
            try:
                result = cleaner.selected_dirs_handler(["selected"])
            finally:
                cleaner.close_log_session()

            self.assertTrue(result)
            self.assertFalse(os.path.exists(root_ds_store))
            self.assertFalse(os.path.exists(nested_ds_store))
            self.assertFalse(os.path.exists(paired_appledouble))
            self.assertTrue(os.path.exists(similarly_named_file))
            self.assertTrue(os.path.exists(normal_file))
            self.assertTrue(os.path.exists(orphan_appledouble))
            self.assertEqual(cleaner.files_found, 3)
            self.assertEqual(cleaner.files_deleted, 3)

    @staticmethod
    def _touch(path: str):
        with open(path, "w", encoding="utf-8") as file:
            file.write("test")


if __name__ == "__main__":
    unittest.main()
