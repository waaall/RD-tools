"""
用法说明:
    1. 运行 `python install.py`
    2. 菜单 `1` 安装当前 Python 的运行依赖
    3. 菜单 `2` 创建或更新隔离的 build venv
    4. 菜单 `3` 使用 build venv 打包可执行程序

也支持非交互命令:
    - `python install.py install-runtime`
    - `python install.py setup-build-env`
    - `python install.py build`

默认行为:
    1. 默认不安装字幕相关的大依赖链, 即 `faster-whisper / torch / tensorboard / ctranslate2 / onnxruntime`
    2. build venv 默认命名为 `.venv-build-<platform>-<arch>`
    3. macOS 的 GUI 打包默认使用 `--onedir --windowed`

如果需要完整打包字幕能力:
    先设置环境变量 `RD_TOOLS_INCLUDE_TRANSCRIPTION=1`, 再运行脚本
    示例: `RD_TOOLS_INCLUDE_TRANSCRIPTION=1 python install.py build`

如需自定义 build venv 目录:
    设置环境变量 `RD_TOOLS_BUILD_VENV`
"""
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from core.task_registry import get_task_specs


ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR / "requirements"
BASE_REQUIREMENTS_FILE = REQUIREMENTS_DIR / "base.txt"
TRANSCRIPTION_REQUIREMENTS_FILE = REQUIREMENTS_DIR / "transcription.txt"
BUILD_REQUIREMENTS_FILE = REQUIREMENTS_DIR / "build.txt"
MAIN_FILE = ROOT_DIR / "main.py"
WINDOWS_OPENH264_DLL = ROOT_DIR / "libs" / "openh264-1.8.0-win64.dll"
APP_NAME = "RD_Tool"
BUILD_VENV_PREFIX = ".venv-build"
BUILD_VENV_ENV = "RD_TOOLS_BUILD_VENV"
PYINSTALLER_CONFIG_DIR = ROOT_DIR / ".pyinstaller-cache"
MPLCONFIG_DIR = ROOT_DIR / ".mplconfig"
INCLUDE_TRANSCRIPTION_ENV = "RD_TOOLS_INCLUDE_TRANSCRIPTION"
DEFAULT_INCLUDE_TRANSCRIPTION = False
TRANSCRIPTION_REQUIREMENTS = ("faster-whisper",)
EXCLUDED_TRANSCRIPTION_MODULES = (
    "faster_whisper",
    "torch",
    "tensorboard",
    "ctranslate2",
    "onnxruntime",
    "av",
    "tokenizers",
    "transformers",
)
RESOURCE_DATA_DIRECTORIES = (
    ("ui/qss", "ui/qss"),
    ("configs", "configs"),
)
CLI_ACTIONS = ("install-runtime", "setup-build-env", "build")


def _env_flag_enabled(env_name: str, default: bool = False) -> bool:
    value = os.environ.get(env_name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _should_include_transcription_stack() -> bool:
    return _env_flag_enabled(
        INCLUDE_TRANSCRIPTION_ENV,
        default=DEFAULT_INCLUDE_TRANSCRIPTION,
    )


def _slugify_label(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "unknown"


def _default_build_venv_dir(
    current_platform: str | None = None,
    machine: str | None = None,
) -> Path:
    resolved_platform = current_platform or platform.system()
    resolved_machine = machine or platform.machine()
    return ROOT_DIR / (
        f"{BUILD_VENV_PREFIX}-"
        f"{_slugify_label(resolved_platform)}-"
        f"{_slugify_label(resolved_machine)}"
    )


def _resolve_build_venv_dir() -> Path:
    override = os.environ.get(BUILD_VENV_ENV)
    if not override:
        return _default_build_venv_dir()

    override_path = Path(override).expanduser()
    if not override_path.is_absolute():
        override_path = ROOT_DIR / override_path
    return override_path


def _venv_python_path(venv_dir: Path, current_platform: str | None = None) -> Path:
    resolved_platform = current_platform or platform.system()
    if resolved_platform == "Windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _pyinstaller_data_separator(current_platform: str) -> str:
    return ";" if current_platform == "Windows" else ":"


def _extend_command_with_resource_data(command: list[str], current_platform: str) -> None:
    separator = _pyinstaller_data_separator(current_platform)
    for source_dir, target_dir in RESOURCE_DATA_DIRECTORIES:
        source_path = ROOT_DIR / source_dir
        command.extend(["--add-data", f"{source_path}{separator}{target_dir}"])


def _registered_task_module_paths() -> list[str]:
    module_paths: list[str] = []
    for spec in get_task_specs():
        # 任务模块是通过注册表里的字符串在运行时动态导入的，
        # PyInstaller 静态分析看不到这类依赖，所以这里统一收集。
        if spec.module_path not in module_paths:
            module_paths.append(spec.module_path)
    return module_paths


def _selected_requirement_files(
    include_transcription: bool,
    include_build_tools: bool = False,
) -> list[Path]:
    requirement_files = [BASE_REQUIREMENTS_FILE]
    if include_transcription:
        requirement_files.append(TRANSCRIPTION_REQUIREMENTS_FILE)
    if include_build_tools:
        requirement_files.append(BUILD_REQUIREMENTS_FILE)
    return requirement_files


def _install_requirement_files(python_executable: Path, requirement_files: list[Path]) -> None:
    missing_files = [str(path) for path in requirement_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing requirement files: " + ", ".join(missing_files)
        )

    command = [str(python_executable), "-m", "pip", "install"]
    for requirement_file in requirement_files:
        command.extend(["-r", str(requirement_file)])
    subprocess.check_call(command)


def _print_transcription_profile(include_transcription: bool, context: str) -> None:
    if include_transcription:
        print(
            f"{context}: including optional transcription dependencies: "
            + ", ".join(TRANSCRIPTION_REQUIREMENTS)
        )
        return

    print(
        f"{context}: skipping optional transcription dependencies: "
        + ", ".join(TRANSCRIPTION_REQUIREMENTS)
    )
    print(
        f"Set {INCLUDE_TRANSCRIPTION_ENV}=1 before running this script "
        "to include faster-whisper/torch-related packaging."
    )


def install_runtime_requirements() -> None:
    include_transcription = _should_include_transcription_stack()
    _print_transcription_profile(include_transcription, "Runtime install")
    requirement_files = _selected_requirement_files(
        include_transcription=include_transcription,
    )
    _install_requirement_files(Path(sys.executable), requirement_files)


def setup_build_env() -> Path:
    include_transcription = _should_include_transcription_stack()
    build_venv_dir = _resolve_build_venv_dir()
    build_python = _venv_python_path(build_venv_dir)

    if build_python.exists():
        print(f"Reusing build venv: {build_venv_dir}")
    else:
        print(f"Creating build venv: {build_venv_dir}")
        subprocess.check_call([sys.executable, "-m", "venv", str(build_venv_dir)])

    # 打包环境必须和当前开发解释器解耦，否则别的项目装进来的包和 hook
    # 也会参与 PyInstaller 分析，重新带回“环境臃肿”和噪音 warning。
    subprocess.check_call([str(build_python), "-m", "pip", "install", "--upgrade", "pip"])
    _print_transcription_profile(include_transcription, "Build env setup")
    requirement_files = _selected_requirement_files(
        include_transcription=include_transcription,
        include_build_tools=True,
    )
    _install_requirement_files(build_python, requirement_files)
    print(f"Build env ready: {build_venv_dir}")
    return build_venv_dir


def _ensure_build_python(current_platform: str) -> Path:
    build_venv_dir = _resolve_build_venv_dir()
    build_python = _venv_python_path(build_venv_dir, current_platform=current_platform)
    if build_python.exists():
        # build 阶段优先复用已准备好的 build venv，
        # 避免每次打包都重新触发 pip 联网安装。
        print(f"Reusing build venv for build: {build_venv_dir}")
        return build_python

    setup_build_env()
    if not build_python.exists():
        raise FileNotFoundError(f"Build env python not found: {build_python}")
    return build_python


def _build_process_env() -> dict[str, str]:
    env = os.environ.copy()
    # 把 PyInstaller 和 Matplotlib 缓存放到仓库内可写目录，
    # 避免用户目录权限和旧缓存影响构建结果。
    env.setdefault("PYINSTALLER_CONFIG_DIR", str(PYINSTALLER_CONFIG_DIR))
    env.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
    return env


def _build_pyinstaller_command(
    python_executable: Path,
    current_platform: str,
    include_transcription_stack: bool,
) -> list[str]:
    command = [
        str(python_executable),
        "-m",
        "PyInstaller",
        "--onedir",
        "--noconfirm",
        "--windowed",
        "--hidden-import=cv2",
    ]

    for module_path in _registered_task_module_paths():
        # 这些任务模块不会在主入口静态 import，
        # 必须显式作为 hidden import 交给 PyInstaller。
        command.append(f"--hidden-import={module_path}")

    if current_platform == "Windows":
        if not WINDOWS_OPENH264_DLL.exists():
            raise FileNotFoundError(f"Missing required DLL: {WINDOWS_OPENH264_DLL}")
        command.extend([
            "--collect-submodules=pydicom",
            "--add-data",
            f"{WINDOWS_OPENH264_DLL};libs",
        ])
    elif current_platform == "Darwin":
        command.append("--collect-submodules=pydicom")
    elif current_platform == "Linux":
        command.extend([
            "--hidden-import=pydicom",
            "--collect-submodules=pydicom",
        ])
    else:
        raise RuntimeError(f"Unsupported operating system: {current_platform}")

    command.append(f"--name={APP_NAME}")
    _extend_command_with_resource_data(command, current_platform)

    if not include_transcription_stack:
        for module_name in EXCLUDED_TRANSCRIPTION_MODULES:
            command.extend(["--exclude-module", module_name])

    command.append(str(MAIN_FILE))
    return command


def build_executable() -> None:
    current_platform = platform.system()
    include_transcription_stack = _should_include_transcription_stack()
    print(f"current_platform: {current_platform}")
    if not include_transcription_stack:
        print(
            "Build profile: excluding optional transcription modules from the bundle: "
            + ", ".join(EXCLUDED_TRANSCRIPTION_MODULES)
        )
        print(
            f"Set {INCLUDE_TRANSCRIPTION_ENV}=1 before running this script "
            "to include faster-whisper/torch-related packaging."
        )

    build_python = _ensure_build_python(current_platform)
    command = _build_pyinstaller_command(
        python_executable=build_python,
        current_platform=current_platform,
        include_transcription_stack=include_transcription_stack,
    )
    subprocess.check_call(command, env=_build_process_env())


def _print_help() -> None:
    print("Usage:")
    print("  python install.py")
    for action in CLI_ACTIONS:
        print(f"  python install.py {action}")


def _run_cli_action(action: str) -> None:
    if action == "install-runtime":
        install_runtime_requirements()
        return
    if action == "setup-build-env":
        setup_build_env()
        return
    if action == "build":
        build_executable()
        return
    raise ValueError(f"Unknown action: {action}")


def _run_interactive_menu() -> None:
    while True:
        print("select wanted function:")
        print("\t1. Install Runtime Requirements")
        print("\t2. Create Or Update Build Env")
        print("\t3. Build Executable (Build Env)")
        print("\t4. Exit")

        try:
            choice = int(input("Enter the corresponding number:"))
            if choice == 1:
                install_runtime_requirements()
            elif choice == 2:
                setup_build_env()
            elif choice == 3:
                build_executable()
            elif choice == 4:
                print("exit")
                break
            else:
                print("\n\tError: Invalid option. Please try again.\n")
        except ValueError:
            print("\n\tError: Invalid input. Please enter a valid number.\n")


def main() -> None:
    if len(sys.argv) > 1:
        action = sys.argv[1].strip().lower()
        if action in {"-h", "--help", "help"}:
            _print_help()
            return
        _run_cli_action(action)
        return

    _run_interactive_menu()


if __name__ == "__main__":
    main()
