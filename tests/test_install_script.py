from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import install


class InstallScriptTests(unittest.TestCase):
    def test_default_build_venv_dir_includes_platform_and_machine(self):
        build_dir = install._default_build_venv_dir(
            current_platform="Darwin",
            machine="arm64",
        )

        self.assertEqual(build_dir, install.ROOT_DIR / ".venv-build-darwin-arm64")

    def test_resolve_build_venv_dir_uses_relative_override(self):
        with patch.dict(os.environ, {install.BUILD_VENV_ENV: "custom-build-env"}):
            build_dir = install._resolve_build_venv_dir()

        self.assertEqual(build_dir, install.ROOT_DIR / "custom-build-env")

    def test_selected_requirement_files_for_build_env_excludes_transcription_by_default(self):
        requirement_files = install._selected_requirement_files(
            include_transcription=False,
            include_build_tools=True,
        )

        self.assertEqual(
            requirement_files,
            [
                install.BASE_REQUIREMENTS_FILE,
                install.BUILD_REQUIREMENTS_FILE,
            ],
        )

    def test_venv_python_path_uses_platform_specific_location(self):
        posix_python = install._venv_python_path(
            Path("/tmp/build-env"),
            current_platform="Darwin",
        )
        windows_python = install._venv_python_path(
            Path("C:/build-env"),
            current_platform="Windows",
        )

        self.assertEqual(posix_python, Path("/tmp/build-env/bin/python"))
        self.assertEqual(windows_python, Path("C:/build-env/Scripts/python.exe"))

    def test_build_pyinstaller_command_uses_build_python_and_excludes_optional_modules(self):
        command = install._build_pyinstaller_command(
            python_executable=Path("/tmp/build-env/bin/python"),
            current_platform="Darwin",
            include_transcription_stack=False,
        )

        self.assertEqual(command[:3], ["/tmp/build-env/bin/python", "-m", "PyInstaller"])
        self.assertIn("--collect-submodules=pydicom", command)
        self.assertIn("--name=RD_Tool", command)
        self.assertIn("ui/qss", " ".join(command))
        self.assertIn("configs", " ".join(command))
        self.assertIn("--exclude-module", command)
        self.assertIn("faster_whisper", command)

    def test_build_executable_runs_pyinstaller_from_build_venv(self):
        build_venv_dir = install.ROOT_DIR / ".venv-build-darwin-arm64"
        expected_python = build_venv_dir / "bin" / "python"

        with patch("install.setup_build_env", return_value=build_venv_dir), patch(
            "install.platform.system",
            return_value="Darwin",
        ), patch(
            "install._build_pyinstaller_command",
            return_value=["pyinstaller-command"],
        ) as mocked_builder, patch(
            "install.subprocess.check_call",
        ) as mocked_check_call:
            install.build_executable()

        mocked_builder.assert_called_once_with(
            python_executable=expected_python,
            current_platform="Darwin",
            include_transcription_stack=False,
        )
        mocked_check_call.assert_called_once_with(["pyinstaller-command"])


if __name__ == "__main__":
    unittest.main()
