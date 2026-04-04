from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.settings_schema import build_default_settings_payload
from modules.app_settings import AppSettings
from ui.task_ui_registry import build_task_descriptors
from widgets.setting_page import SettingWindow


class SettingPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

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

    def test_warning_banner_is_visible_when_user_config_is_invalid(self):
        os.makedirs(os.path.dirname(self.user_settings_file), exist_ok=True)
        with open(self.user_settings_file, "w", encoding="utf-8") as fh:
            fh.write("{bad json")

        settings = AppSettings()
        window = SettingWindow(settings, build_task_descriptors())

        self.assertFalse(window.config_health_card.isHidden())
        self.assertIn("用户配置", window.config_health_body.text())
        self.assertTrue(window.has_task_settings("files-renamer"))
        window.deleteLater()

    def test_reset_defaults_button_rebuilds_user_settings_file(self):
        os.makedirs(os.path.dirname(self.user_settings_file), exist_ok=True)
        with open(self.user_settings_file, "w", encoding="utf-8") as fh:
            fh.write("{bad json")

        settings = AppSettings()
        window = SettingWindow(settings, build_task_descriptors())

        with patch("widgets.setting_page.MessageBox.exec", return_value=1):
            window._reset_user_settings_to_defaults()

        with open(self.user_settings_file, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        self.assertEqual(payload, build_default_settings_payload())
        self.assertFalse(window.settings.get_config_health().has_issues)
        window.deleteLater()
