from __future__ import annotations

import os
import tempfile
import unittest

from modules.app_settings import AppSettings
from modules.files_basic import FilesBasic
from core.task_params import build_task_params
from core.task_registry import TaskSpec


class DummyTask(FilesBasic):
    def __init__(
        self,
        foo: str = "ctor-default",
        bar: str = "ctor-bar",
        log_folder_name: str = "dummy-log",
    ):
        super().__init__(log_folder_name=log_folder_name, out_dir_prefix="dummy-out-", parallel=False)
        self.foo = foo
        self.bar = bar


class RequiredTask(FilesBasic):
    def __init__(self, required_value: str, log_folder_name: str = "required-log"):
        super().__init__(log_folder_name=log_folder_name, out_dir_prefix="required-out-", parallel=False)
        self.required_value = required_value


class FakeSettings:
    def __init__(self, group_values: dict[str, object] | None = None):
        self._group_values = dict(group_values or {})

    def get_group_values(self, category_name: str, group_name: str) -> dict[str, object]:
        if category_name != "Batch_Files":
            return {}
        return dict(self._group_values)


class TaskParamsTests(unittest.TestCase):
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
        AppSettings()

    def tearDown(self):
        if self._home_backup is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = self._home_backup
        self._temp_home.cleanup()

    def test_get_group_values_reads_batch_file_group_by_task_key(self):
        settings = AppSettings()

        values = settings.get_group_values("Batch_Files", "files-renamer")

        self.assertEqual(values["mode"], "prefix")
        self.assertIn("pattern", values)
        self.assertIn("max_threads", values)

    def test_get_setting_entries_filters_by_task_key_group(self):
        settings = AppSettings()

        entries = settings.get_setting_entries("Batch_Files", group_name="files-renamer")

        self.assertTrue(entries)
        self.assertTrue(all(entry["path"][1] == "files-renamer" for entry in entries))
        self.assertIn("mode", {entry["path"][-1] for entry in entries})

    def test_build_task_params_uses_task_key_group_and_default_params(self):
        settings = FakeSettings(
            {
                "foo": "from-config",
                "log_folder_name": "from-settings",
            }
        )
        task_spec = TaskSpec(
            key="dummy-task",
            title="Dummy",
            description="",
            module_path="modules.dummy_task",
            class_name="DummyTask",
            default_params={"foo": "from-default", "bar": "from-spec"},
        )

        params = build_task_params(task_spec, settings, DummyTask)

        self.assertEqual(
            params,
            {
                "foo": "from-config",
                "bar": "from-spec",
                "log_folder_name": "from-settings",
            },
        )

    def test_build_task_params_rejects_unknown_keys(self):
        settings = FakeSettings({"unknown_value": "bad"})
        task_spec = TaskSpec(
            key="dummy-task",
            title="Dummy",
            description="",
            module_path="modules.dummy_task",
            class_name="DummyTask",
        )

        with self.assertRaisesRegex(ValueError, "未知参数"):
            build_task_params(task_spec, settings, DummyTask)

    def test_build_task_params_rejects_missing_required_params(self):
        settings = FakeSettings()
        task_spec = TaskSpec(
            key="required-task",
            title="Required",
            description="",
            module_path="modules.required_task",
            class_name="RequiredTask",
        )

        with self.assertRaisesRegex(ValueError, "缺少必填参数"):
            build_task_params(task_spec, settings, RequiredTask)

    def test_build_task_params_lets_ctor_defaults_fill_remaining_values(self):
        settings = FakeSettings()
        task_spec = TaskSpec(
            key="dummy-task",
            title="Dummy",
            description="",
            module_path="modules.dummy_task",
            class_name="DummyTask",
            default_params={"foo": "from-spec"},
        )

        params = build_task_params(task_spec, settings, DummyTask)

        self.assertEqual(params, {"foo": "from-spec"})


if __name__ == "__main__":
    unittest.main()
