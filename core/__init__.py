from .task_message import MessageLevel, TaskMessage, ensure_task_message

__all__ = [
    'MessageLevel',
    'TaskErrorCode',
    'TaskLoader',
    'TaskMessage',
    'TaskUserError',
    'ensure_task_message',
    'format_task_exception',
]


def __getattr__(name: str):
    # Keep ``import core.task_registry`` lightweight for install.py before
    # runtime dependencies such as PySide6 have been installed.
    if name == 'TaskLoader':
        from .task_loader import TaskLoader

        return TaskLoader
    if name in {'TaskErrorCode', 'TaskUserError', 'format_task_exception'}:
        from .task_errors import TaskErrorCode, TaskUserError, format_task_exception

        return {
            'TaskErrorCode': TaskErrorCode,
            'TaskUserError': TaskUserError,
            'format_task_exception': format_task_exception,
        }[name]
    raise AttributeError(f"module 'core' has no attribute {name!r}")
