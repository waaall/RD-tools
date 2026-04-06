"""
用法说明:
    1. 运行 `python install.py`
    2. 菜单 `1` 安装依赖
    3. 菜单 `2` 打包可执行程序

默认行为:
    1. 默认不安装字幕相关的大依赖链, 即 `faster_whisper / torch / tensorboard / ctranslate2 / onnxruntime`
    2. 打包时默认也会排除这条依赖链, 以减少 warning、体积和构建时间
    3. macOS 的 GUI 打包默认使用 `--onedir --windowed`

如果需要完整打包字幕能力:
    先设置环境变量 `RD_TOOLS_INCLUDE_TRANSCRIPTION=1`, 再运行脚本
    示例: `RD_TOOLS_INCLUDE_TRANSCRIPTION=1 python install.py`

已知说明:
    1. Windows 平台打包 `opencv-python` 时可能缺少额外 dll
    2. `pydicom` 的依赖不会总被自动收集, 所以这里显式交给 PyInstaller 处理
    3. Windows 若缺少 openh264, 需要额外准备 `libs/openh264-1.8.0-win64.dll`
"""
import os
import re
import subprocess
import platform
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_FILE = ROOT_DIR / "requirements.txt"
MAIN_FILE = ROOT_DIR / "main.py"
WINDOWS_OPENH264_DLL = ROOT_DIR / "libs" / "openh264-1.8.0-win64.dll"
APP_NAME = "RD_Tool"
INCLUDE_TRANSCRIPTION_ENV = "RD_TOOLS_INCLUDE_TRANSCRIPTION"
DEFAULT_INCLUDE_TRANSCRIPTION = False
OPTIONAL_REQUIREMENTS = {"faster_whisper"}
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


def _normalize_requirement_name(requirement: str) -> str:
    package_name = re.split(r"[<>=!~;\[\s]", requirement, maxsplit=1)[0]
    return package_name.strip().lower().replace("-", "_")


def _read_requirements(requirements_file: Path) -> list[str]:
    requirements: list[str] = []
    for raw_line in requirements_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


def _resolve_requirements_for_current_platform() -> list[str]:
    requirements = _read_requirements(REQUIREMENTS_FILE)
    if _should_include_transcription_stack():
        return requirements

    filtered_requirements = [
        requirement
        for requirement in requirements
        if _normalize_requirement_name(requirement) not in OPTIONAL_REQUIREMENTS
    ]
    skipped_requirements = [
        requirement
        for requirement in requirements
        if _normalize_requirement_name(requirement) in OPTIONAL_REQUIREMENTS
    ]
    if skipped_requirements:
        print(
            "Default profile: skipping optional transcription dependencies: "
            + ", ".join(skipped_requirements)
        )
        print(
            f"Set {INCLUDE_TRANSCRIPTION_ENV}=1 before running this script "
            "to include faster-whisper/torch-related packaging."
        )
    return filtered_requirements


def install_requirements():
    requirements = _resolve_requirements_for_current_platform()
    subprocess.check_call([sys.executable, "-m", "pip", "install", *requirements])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyInstaller"])


def build_executable():
    # 检测操作系统
    current_platform = platform.system()
    include_transcription_stack = _should_include_transcription_stack()
    print(f"current_platform: {current_platform}")
    if not include_transcription_stack:
        print(
            "Default profile: excluding optional transcription modules from the bundle: "
            + ", ".join(EXCLUDED_TRANSCRIPTION_MODULES)
        )
        print(
            f"Set {INCLUDE_TRANSCRIPTION_ENV}=1 before running this script "
            "to include faster-whisper/torch-related packaging."
        )

    if current_platform == "Windows":
        # pyinstaller --onedir --windowed --collect-submodules=pydicom --hidden-import=cv2 --add-data "libs/openh264-1.8.0-win64.dll;libs" --name=RD_Tool main.py
        # Windows 平台的 PyInstaller 命令 # "--add-data", "libs/openh264-1.8.0-win64.dll;.",  # dll添加到 exe 同一目录
        if not WINDOWS_OPENH264_DLL.exists():
            raise FileNotFoundError(f"Missing required DLL: {WINDOWS_OPENH264_DLL}")
        command = [sys.executable, "-m", "PyInstaller",
                   "--onedir",  # 生成一个文件夹,内部包含所有运行所需文件
                   "--windowed",
                   "--hidden-import=cv2",
                   "--collect-submodules=pydicom", # 确保 pydicom 所需模块被包含
                   "--add-data", f"{WINDOWS_OPENH264_DLL};libs",  # dll添加到 libs 文件夹
                   f"--name={APP_NAME}"]
        if not include_transcription_stack:
            for module_name in EXCLUDED_TRANSCRIPTION_MODULES:
                command.extend(["--exclude-module", module_name])
        command.append(str(MAIN_FILE))
        subprocess.check_call(command)
    elif current_platform == "Darwin":
        command = [sys.executable, "-m", "PyInstaller",
                   "--onedir",  # macOS GUI 应用优先使用 onedir，避免 onefile 与 .app 的冲突
                   "--windowed",
                   "--hidden-import=cv2",
                   "--collect-submodules=pydicom", # 确保 pydicom 所需模块被包含
                   f"--name={APP_NAME}"]
        if not include_transcription_stack:
            for module_name in EXCLUDED_TRANSCRIPTION_MODULES:
                command.extend(["--exclude-module", module_name])
        command.append(str(MAIN_FILE))
        subprocess.check_call(command)
    elif current_platform == "Linux":
        command = [sys.executable, "-m", "PyInstaller",
                   "--onedir",
                   "--windowed",
                   "--hidden-import=pydicom",
                   "--hidden-import=cv2",
                   "--collect-submodules=pydicom",
                   f"--name={APP_NAME}"]
        if not include_transcription_stack:
            for module_name in EXCLUDED_TRANSCRIPTION_MODULES:
                command.extend(["--exclude-module", module_name])
        command.append(str(MAIN_FILE))
        subprocess.check_call(command)
    else:
        raise RuntimeError(f"Unsupported operating system: {current_platform}")


def main() -> None:
    while True:
        print("select wanted function:")
        print("\t1. Install Requirements")
        print("\t2. Build Executable")
        print("\t3. Exit")

        try:
            choice = int(input("Enter the corresponding number:"))
            if choice == 1:
                install_requirements()
            elif choice == 2:
                build_executable()
            elif choice == 3:
                print("exit")
                break
            else:
                print("\n\tError: Invalid option. Please try again.\n")
        except ValueError:
            print("\n\tError: Invalid input. Please enter a valid number.\n")


if __name__ == "__main__":
    main()
