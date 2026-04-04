from __future__ import annotations

import os
import sys

from core import MessageLevel, TaskLoader
from core.task_params import build_task_params
from core.task_registry import get_task_spec
from modules.app_settings import AppSettings


def _prompt_work_folder(task_title: str) -> str:
    print(f'运行任务: {task_title}')
    input_path = input('请输入工作目录绝对路径(直接回车则使用当前目录):\n').strip()
    if not input_path:
        return os.getcwd()
    if os.path.isdir(input_path):
        return os.path.abspath(input_path)
    raise ValueError(f'路径 {input_path} 不存在或不是文件夹。')


def _prompt_selected_dirs(possible_dirs: list[str]) -> list[str]:
    if not possible_dirs:
        raise ValueError('当前工作目录下没有可处理的一级子目录。')

    print('\n可处理的子目录:')
    for index, folder_name in enumerate(possible_dirs):
        print(f'{index}: {folder_name}')

    user_input = input('\n请选择要处理的序号(用空格分隔多个序号):\n').strip()
    if not user_input:
        raise ValueError('至少选择一个待处理目录。')

    try:
        raw_indices = [int(token) for token in user_input.split()]
    except ValueError as exc:
        raise ValueError('输入错误, 必须输入数字。') from exc

    selected_dirs: list[str] = []
    seen_indices: set[int] = set()
    for index in raw_indices:
        if index < 0 or index >= len(possible_dirs):
            raise ValueError('输入数字不在提供范围, 请重新运行。')
        if index in seen_indices:
            continue
        seen_indices.add(index)
        selected_dirs.append(possible_dirs[index])

    return selected_dirs


def _report_cli_error(handler, message: str):
    if handler is None:
        print(message, file=sys.stderr)
        return
    handler.send_message(message, level=MessageLevel.ERROR)


def run_task_cli(task_key: str, operation_cls: type | None = None) -> int:
    handler = None
    try:
        task_spec = get_task_spec(task_key)
        settings = AppSettings()
        settings.load_settings()

        resolved_operation_cls = operation_cls or TaskLoader.load_class(task_spec.module_path, task_spec.class_name)
        params = build_task_params(task_spec, settings, resolved_operation_cls)
        handler = resolved_operation_cls(**params)

        work_folder = _prompt_work_folder(task_spec.title)
        handler.set_work_folder(work_folder)
        selected_dirs = _prompt_selected_dirs(handler.possble_dirs or [])
        if not handler.selected_dirs_handler(selected_dirs):
            _report_cli_error(handler, '任务未成功完成。')
            return 1
        return 0
    except Exception as exc:
        _report_cli_error(handler, f'CLI 执行失败: {exc}')
        return 1
    finally:
        if handler is not None:
            handler.close_log_session()
