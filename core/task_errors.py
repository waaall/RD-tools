"""任务层用户可见错误的结构化标记与翻译入口。

底层模块只抛出稳定 code + 参数 + 原始 detail；UI/任务框架在展示时再按当前
translator 翻译标题和前缀，避免在任务层硬编码中文或提前绑定语言。
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from PySide6.QtCore import QCoreApplication, QT_TRANSLATE_NOOP


TASK_ERROR_CONTEXT = 'TaskErrors'


class TaskErrorCode(str, Enum):
    """任务框架可识别的用户可见错误类型。"""

    MODULE_LOAD_FAILED = 'module_load_failed'
    CLASS_NOT_FOUND = 'class_not_found'
    INVALID_TASK_CLASS_BASE = 'invalid_task_class_base'
    UNKNOWN_PARAMS = 'unknown_params'
    MISSING_PARAMS = 'missing_params'
    UNKNOWN = 'unknown'


# 用 NOOP 保存英文源串，真正显示时再按当前语言 translate。
_TASK_ERROR_TEMPLATES = {
    TaskErrorCode.MODULE_LOAD_FAILED: QT_TRANSLATE_NOOP(
        'TaskErrors',
        'Could not load task module {module_path}.',
    ),
    TaskErrorCode.CLASS_NOT_FOUND: QT_TRANSLATE_NOOP(
        'TaskErrors',
        'Task module {module_path} does not define class {class_name}.',
    ),
    TaskErrorCode.INVALID_TASK_CLASS_BASE: QT_TRANSLATE_NOOP(
        'TaskErrors',
        'Task {task_key} must inherit FilesBasic.',
    ),
    TaskErrorCode.UNKNOWN_PARAMS: QT_TRANSLATE_NOOP(
        'TaskErrors',
        'Task {task_key} has unknown config parameters: {params}.',
    ),
    TaskErrorCode.MISSING_PARAMS: QT_TRANSLATE_NOOP(
        'TaskErrors',
        'Task {task_key} is missing required parameters: {params}.',
    ),
    TaskErrorCode.UNKNOWN: QT_TRANSLATE_NOOP(
        'TaskErrors',
        'Task error.',
    ),
}


class TaskUserError(Exception):
    """携带稳定错误标记的任务层异常。

    detail 原样保留，用于展示底层异常信息；不要把 detail 当作待翻译内容。
    """

    def __init__(
        self,
        code: TaskErrorCode | str,
        params: dict[str, Any] | None = None,
        *,
        detail: str | None = None,
    ) -> None:
        self.code = TaskErrorCode(code)
        self.params = dict(params or {})
        self.detail = detail
        super().__init__(self.code.value)

    def __str__(self) -> str:
        return self.to_user_message()

    def to_user_message(self) -> str:
        """把错误 code 翻译成用户可见文字，detail 保持原文。"""
        source = _TASK_ERROR_TEMPLATES.get(self.code, _TASK_ERROR_TEMPLATES[TaskErrorCode.UNKNOWN])
        message = QCoreApplication.translate(TASK_ERROR_CONTEXT, source).format(**self.params)
        if not self.detail:
            return message
        detail_text = QCoreApplication.translate('TaskErrors', 'Detail: {0}').format(self.detail)
        return f'{message} {detail_text}'


def format_task_exception(exc: Exception) -> str:
    """任务框架统一调用的异常格式化入口。"""
    if isinstance(exc, TaskUserError):
        return exc.to_user_message()
    return str(exc)
