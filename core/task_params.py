from __future__ import annotations

import inspect

from core.task_errors import TaskErrorCode, TaskUserError
from modules.app_settings import AppSettings
from modules.files_basic import FilesBasic
from core.task_registry import TaskSpec


def build_task_params(
    task_spec: TaskSpec,
    settings: AppSettings,
    operation_cls: type,
    overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    try:
        if not issubclass(operation_cls, FilesBasic):
            raise TypeError
    except TypeError as exc:
        raise TaskUserError(
            TaskErrorCode.INVALID_TASK_CLASS_BASE,
            {'task_key': task_spec.key},
            detail=str(exc) or None,
        ) from exc

    resolved_params: dict[str, object] = dict(task_spec.default_params)
    for key, value in settings.get_group_values('Batch_Files', task_spec.key).items():
        if value is None:
            continue
        resolved_params[key] = value
    if overrides:
        resolved_params.update(overrides)

    signature = inspect.signature(operation_cls.__init__)
    accepted_params: set[str] = set()
    required_params: set[str] = set()
    accepts_var_kwargs = False
    for name, parameter in signature.parameters.items():
        if name == 'self':
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_var_kwargs = True
            continue
        if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            continue
        accepted_params.add(name)
        if parameter.default is inspect.Parameter.empty:
            required_params.add(name)

    unknown_params = sorted(key for key in resolved_params if key not in accepted_params)
    if unknown_params and not accepts_var_kwargs:
        raise TaskUserError(
            TaskErrorCode.UNKNOWN_PARAMS,
            {'task_key': task_spec.key, 'params': ', '.join(unknown_params)},
        )

    missing_params = sorted(name for name in required_params if name not in resolved_params)
    if missing_params:
        raise TaskUserError(
            TaskErrorCode.MISSING_PARAMS,
            {'task_key': task_spec.key, 'params': ', '.join(missing_params)},
        )

    return resolved_params
