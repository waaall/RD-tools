from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class MessageLevel(str, Enum):
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    SUCCESS = 'success'

    @classmethod
    def infer(cls, text: str) -> 'MessageLevel':
        normalized = text.strip().lower()
        if normalized.startswith('traceback (most recent call last):'):
            return cls.ERROR
        if normalized.startswith(('error:', 'Error', 'error', '错误')):
            return cls.ERROR
        if normalized.startswith(('warning:', 'Warning', 'warning', '警告', '注意')):
            return cls.WARNING
        if normalized.startswith(('success:', 'Success', 'success', '成功')):
            return cls.SUCCESS
        if any(keyword in normalized for keyword in ('出错', '失败', '异常', '无法')):
            return cls.ERROR
        return cls.INFO


@dataclass(frozen=True)
class TaskMessage:
    text: str
    level: MessageLevel = MessageLevel.INFO
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def build(cls, text: str, level: MessageLevel | str | None = None) -> 'TaskMessage':
        if isinstance(level, str):
            level = MessageLevel(level.lower())
        resolved_level = level or MessageLevel.infer(text)
        normalized_text = cls._strip_level_prefix(str(text), resolved_level)
        return cls(text=normalized_text, level=resolved_level)

    def to_log_lines(self) -> list[str]:
        timestamp = self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        prefix = f'[{timestamp}] {self.level.value.upper():7} '
        lines = self.text.splitlines() or ['']
        formatted_lines = [f'{prefix}{lines[0]}']
        formatted_lines.extend(f'{" " * len(prefix)}{line}' for line in lines[1:])
        return formatted_lines

    @staticmethod
    def _strip_level_prefix(text: str, level: MessageLevel) -> str:
        stripped_text = text.lstrip()
        normalized = stripped_text.lower()
        prefix_map = {
            MessageLevel.ERROR: ('error:', 'error：', '错误:', '错误：'),
            MessageLevel.WARNING: ('warning:', 'warning：', '警告:', '警告：', '注意:', '注意：'),
            MessageLevel.SUCCESS: ('success:', 'success!', '成功:', '成功：'),
        }
        for prefix in prefix_map.get(level, ()):
            if normalized.startswith(prefix):
                return stripped_text[len(prefix):].lstrip()
        return text


def ensure_task_message(message: TaskMessage | str, level: MessageLevel | str | None = None) -> TaskMessage:
    if isinstance(message, TaskMessage):
        return message
    return TaskMessage.build(message, level=level)
