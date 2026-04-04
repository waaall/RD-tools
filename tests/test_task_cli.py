from __future__ import annotations

import os
import runpy
import tempfile
import unittest
from unittest.mock import patch

import core.task_cli as task_cli_module
from core.task_cli import run_task_cli
from core.task_registry import TaskSpec
from modules.app_settings import AppSettings
from modules.files_basic import FilesBasic


class CliSuccessTask(FilesBasic):
    last_instance = None

    def __init__(self, log_folder_name: str = "cli-log", extra_value: str = "ctor-default"):
        super().__init__(log_folder_name=log_folder_name, out_dir_prefix="cli-out-", parallel=False)
        self.extra_value = extra_value
        self.received_dirs: list[str] = []
        type(self).last_instance = self

    def selected_dirs_handler(self, indexs_list):
        self.received_dirs = list(indexs_list)
        self.send_message("CLI task completed.")
        return True


class CliFailTask(CliSuccessTask):
    def selected_dirs_handler(self, indexs_list):
        self.received_dirs = list(indexs_list)
        self.send_message("CLI task failed.")
        return False


class TaskCliTests(unittest.TestCase):
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
        CliSuccessTask.last_instance = None

    def tearDown(self):
        if self._home_backup is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = self._home_backup
        self._temp_home.cleanup()

    def test_run_task_cli_success_returns_zero(self):
        spec = self._dummy_spec()

        with tempfile.TemporaryDirectory() as work_dir:
            os.mkdir(os.path.join(work_dir, "alpha"))
            with patch("core.task_cli.get_task_spec", return_value=spec), patch(
                "core.task_cli.build_task_params",
                return_value={"extra_value": "from-settings"},
            ), patch(
                "builtins.input",
                side_effect=[work_dir, "0"],
            ):
                exit_code = run_task_cli("dummy-task", operation_cls=CliSuccessTask)

        self.assertEqual(exit_code, 0)
        self.assertIsNotNone(CliSuccessTask.last_instance)
        self.assertEqual(CliSuccessTask.last_instance.extra_value, "from-settings")
        self.assertEqual(CliSuccessTask.last_instance.received_dirs, ["alpha"])

    def test_run_task_cli_rejects_invalid_work_directory(self):
        spec = self._dummy_spec()

        with patch("core.task_cli.get_task_spec", return_value=spec), patch(
            "builtins.input",
            side_effect=["/path/that/does/not/exist"],
        ):
            exit_code = run_task_cli("dummy-task", operation_cls=CliSuccessTask)

        self.assertNotEqual(exit_code, 0)

    def test_run_task_cli_rejects_invalid_selection(self):
        spec = self._dummy_spec()

        with tempfile.TemporaryDirectory() as work_dir:
            os.mkdir(os.path.join(work_dir, "alpha"))
            with patch("core.task_cli.get_task_spec", return_value=spec), patch(
                "builtins.input",
                side_effect=[work_dir, "9"],
            ):
                exit_code = run_task_cli("dummy-task", operation_cls=CliSuccessTask)

        self.assertNotEqual(exit_code, 0)

    def test_run_task_cli_uses_shared_param_builder(self):
        spec = self._dummy_spec()

        with tempfile.TemporaryDirectory() as work_dir:
            os.mkdir(os.path.join(work_dir, "alpha"))
            with patch("core.task_cli.get_task_spec", return_value=spec), patch(
                "core.task_cli.build_task_params",
                wraps=task_cli_module.build_task_params,
            ) as mocked_builder, patch(
                "builtins.input",
                side_effect=[work_dir, "0"],
            ):
                exit_code = run_task_cli("dummy-task", operation_cls=CliSuccessTask)

        self.assertEqual(exit_code, 0)
        mocked_builder.assert_called_once()
        called_spec, _, called_cls = mocked_builder.call_args[0]
        self.assertEqual(called_spec.key, "dummy-task")
        self.assertIs(called_cls, CliSuccessTask)

    def test_run_task_cli_returns_non_zero_when_task_reports_failure(self):
        spec = self._dummy_spec()

        with tempfile.TemporaryDirectory() as work_dir:
            os.mkdir(os.path.join(work_dir, "alpha"))
            with patch("core.task_cli.get_task_spec", return_value=spec), patch(
                "builtins.input",
                side_effect=[work_dir, "0"],
            ):
                exit_code = run_task_cli("dummy-task", operation_cls=CliFailTask)

        self.assertNotEqual(exit_code, 0)

    def test_module_main_delegates_to_shared_runner(self):
        with patch("core.task_cli.run_task_cli", return_value=0) as mocked_runner:
            with self.assertRaises(SystemExit) as exc:
                runpy.run_module("modules.mac_poop_scooper", run_name="__main__")

        self.assertEqual(exc.exception.code, 0)
        mocked_runner.assert_called_once()
        self.assertEqual(mocked_runner.call_args.args[0], "mac-cleaner")
        self.assertIn("operation_cls", mocked_runner.call_args.kwargs)

    def _dummy_spec(self):
        return TaskSpec(
            key="dummy-task",
            title="Dummy Task",
            description="",
            module_path="modules.dummy_task",
            class_name="DummyTask",
            default_params={"extra_value": "from-spec"},
        )


if __name__ == "__main__":
    unittest.main()
