from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QCoreApplication

from core.task_registry import TaskSpec


@dataclass(frozen=True, slots=True)
class TaskDescriptor:
    task_spec: TaskSpec
    icon: Any

    @classmethod
    def from_spec(cls, task_spec: TaskSpec, icon: Any) -> 'TaskDescriptor':
        return cls(task_spec=task_spec, icon=icon)

    @property
    def key(self) -> str:
        return self.task_spec.key

    @property
    def title(self) -> str:
        # 任务标题是模块级数据,显示时按当前语言翻译
        return QCoreApplication.translate('Tasks', self.task_spec.title)

    @property
    def description(self) -> str:
        return QCoreApplication.translate('Tasks', self.task_spec.description)

    @property
    def module_path(self) -> str:
        return self.task_spec.module_path

    @property
    def class_name(self) -> str:
        return self.task_spec.class_name

    @property
    def default_params(self) -> dict[str, Any]:
        return self.task_spec.default_params
