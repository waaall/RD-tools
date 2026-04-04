from __future__ import annotations

import json
import os
import tempfile
import unittest

from core.settings_schema import build_default_settings_payload
from modules.app_settings import AppSettings


class AppSettingsWithCustomSnapshot(AppSettings):
    snapshot_path_override: str | None = None

    def _resolve_default_settings_file(self) -> str:
        if self.snapshot_path_override is not None:
            return self.snapshot_path_override
        return super()._resolve_default_settings_file()


class SettingsSchemaTests(unittest.TestCase):
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
        AppSettingsWithCustomSnapshot.snapshot_path_override = None
        self._temp_home.cleanup()

    def test_schema_generated_payload_matches_repo_snapshot(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        snapshot_path = os.path.join(repo_root, "configs", "settings.json")
        with open(snapshot_path, "r", encoding="utf-8") as fh:
            snapshot_payload = json.load(fh)

        self.assertEqual(build_default_settings_payload(), snapshot_payload)

    def test_invalid_user_json_falls_back_to_schema_defaults_and_records_issue(self):
        os.makedirs(os.path.dirname(self.user_settings_file), exist_ok=True)
        with open(self.user_settings_file, "w", encoding="utf-8") as fh:
            fh.write("{bad json")

        settings = AppSettings()

        self.assertEqual(settings.theme, "Auto")
        self.assertEqual(settings.get_group_values("Batch_Files", "files-renamer")["mode"], "prefix")
        issues = settings.get_config_health().issues
        self.assertTrue(any(issue.source == "用户配置" and issue.code == "parse_error" for issue in issues))

    def test_invalid_snapshot_json_uses_schema_defaults_and_records_issue(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as fh:
            fh.write("{bad json")
            snapshot_path = fh.name

        try:
            AppSettingsWithCustomSnapshot.snapshot_path_override = snapshot_path
            settings = AppSettingsWithCustomSnapshot()
        finally:
            os.remove(snapshot_path)

        self.assertEqual(settings.theme, "Auto")
        issues = settings.get_config_health().issues
        self.assertTrue(any(issue.source == "仓库默认配置" and issue.code == "parse_error" for issue in issues))

    def test_legacy_task_group_is_ignored_and_recorded(self):
        settings = AppSettings()
        with open(settings.settings_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        data.setdefault("Batch_Files", {})["DicomToImage"] = {
            "log_folder_name": "legacy-log",
            "fps": 99,
        }
        with open(settings.settings_file, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4)

        settings = AppSettings()

        self.assertEqual(
            settings.get_group_values("Batch_Files", "dicom-processing")["log_folder_name"],
            "dicom_handle_log",
        )
        self.assertTrue(any("DicomToImage" in issue.message for issue in settings.get_config_health().issues))

    def test_successful_save_rewrites_user_file_to_normalized_payload(self):
        settings = AppSettings()
        with open(settings.settings_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        data.setdefault("Batch_Files", {})["DicomToImage"] = {"log_folder_name": "legacy-log"}
        with open(settings.settings_file, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4)

        settings = AppSettings()
        self.assertTrue(settings.save_settings("theme", "Light"))

        with open(settings.settings_file, "r", encoding="utf-8") as fh:
            normalized = json.load(fh)

        self.assertNotIn("DicomToImage", normalized.get("Batch_Files", {}))
        self.assertEqual(normalized["General"]["Display"]["theme"], "Light")
