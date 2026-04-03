from __future__ import annotations

import json
import os
import tempfile
import unittest

from modules.app_settings import AppSettings


class AppSettingsTaskOrderTests(unittest.TestCase):
    def setUp(self):
        self._home_backup = os.environ.get("HOME")
        self._temp_home = tempfile.TemporaryDirectory()
        os.environ["HOME"] = self._temp_home.name
        self.user_settings_file = os.path.join(
            self._temp_home.name,
            "Develop",
            "RD-tools-configs",
            "settings.json",
        )

    def tearDown(self):
        if self._home_backup is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = self._home_backup
        self._temp_home.cleanup()

    def test_default_task_order_matches_product_spec(self):
        self.assertEqual(
            AppSettings.DEFAULT_TASK_ORDER,
            [
                "files-renamer",
                "bilibili-export",
                "subtitle-generation",
                "mac-cleaner",
                "merge-colors",
                "split-colors",
                "twist-images",
                "ecg-handler",
                "dicom-processing",
            ],
        )

    def test_missing_task_center_silently_falls_back_to_default_order(self):
        settings = self._build_settings(lambda data: data.pop("Task_Center", None))

        ordered_keys = settings.get_task_order([
            *AppSettings.DEFAULT_TASK_ORDER,
            "new-task",
        ])

        self.assertEqual(
            ordered_keys,
            [
                *AppSettings.DEFAULT_TASK_ORDER,
                "new-task",
            ],
        )
        self.assertEqual(settings.consume_startup_warnings(), [])

    def test_missing_task_order_silently_falls_back_to_default_order(self):
        settings = self._build_settings(lambda data: data.setdefault("Task_Center", {}).pop("task_order", None))

        ordered_keys = settings.get_task_order(AppSettings.DEFAULT_TASK_ORDER)

        self.assertEqual(ordered_keys, AppSettings.DEFAULT_TASK_ORDER)
        self.assertEqual(settings.consume_startup_warnings(), [])

    def test_non_list_task_order_falls_back_and_records_warning_once(self):
        settings = self._build_settings(
            lambda data: data.setdefault("Task_Center", {}).update({"task_order": "bad-value"})
        )

        ordered_keys = settings.get_task_order(AppSettings.DEFAULT_TASK_ORDER)

        self.assertEqual(ordered_keys, AppSettings.DEFAULT_TASK_ORDER)
        warnings = settings.consume_startup_warnings()
        self.assertEqual(len(warnings), 1)
        self.assertIn("配置非法", warnings[0])
        self.assertEqual(settings.consume_startup_warnings(), [])

    def test_invalid_items_are_filtered_and_remaining_tasks_are_appended(self):
        def mutate(data):
            data.setdefault("Task_Center", {})["task_order"] = [
                "subtitle-generation",
                "unknown-task",
                123,
                "subtitle-generation",
                "files-renamer",
            ]

        settings = self._build_settings(mutate)
        available_keys = [
            "merge-colors",
            "files-renamer",
            "subtitle-generation",
            "dicom-processing",
        ]

        ordered_keys = settings.get_task_order(available_keys)

        self.assertEqual(
            ordered_keys,
            [
                "subtitle-generation",
                "files-renamer",
                "merge-colors",
                "dicom-processing",
            ],
        )
        warnings = settings.consume_startup_warnings()
        self.assertEqual(len(warnings), 1)
        self.assertIn("已过滤非法项并补齐缺失任务", warnings[0])

    def test_partial_task_order_appends_unspecified_tasks_in_available_order(self):
        settings = self._build_settings(
            lambda data: data.setdefault("Task_Center", {}).update({"task_order": ["files-renamer"]})
        )
        available_keys = [
            "merge-colors",
            "dicom-processing",
            "files-renamer",
            "new-task",
        ]

        ordered_keys = settings.get_task_order(available_keys)

        self.assertEqual(
            ordered_keys,
            [
                "files-renamer",
                "merge-colors",
                "dicom-processing",
                "new-task",
            ],
        )
        self.assertEqual(settings.consume_startup_warnings(), [])

    def test_save_task_order_failure_does_not_pollute_in_memory_state(self):
        settings = self._build_settings()
        original_order = list(settings.task_order)
        original_json_value = settings.get_value_from_path(("Task_Center", "task_order"))
        settings._write_settings_file = lambda *_args, **_kwargs: False

        saved = settings.save_task_order(["dicom-processing", "files-renamer"])

        self.assertFalse(saved)
        self.assertEqual(settings.task_order, original_order)
        self.assertEqual(
            settings.get_value_from_path(("Task_Center", "task_order")),
            original_json_value,
        )

    def _build_settings(self, mutate_json=None):
        settings = AppSettings()
        if mutate_json is None:
            return settings

        with open(settings.settings_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        mutate_json(data)
        with open(self.user_settings_file, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4)
        return AppSettings()


if __name__ == "__main__":
    unittest.main()
