from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

from core.task_registry import TaskSettingSpec, get_task_specs


@dataclass(frozen=True, slots=True)
class SettingFieldSpec:
    field_id: str
    category: str
    json_key: str
    default: Any
    value_type: type[Any]
    group_key: str | None = None
    options: tuple[Any, ...] | None = None
    label: str | None = None
    description: str | None = None
    visible: bool = True
    coerce: Callable[[Any], Any] | None = field(default=None, repr=False, compare=False)

    @property
    def path(self) -> tuple[str, ...]:
        if self.group_key is None:
            return self.category, self.json_key
        return self.category, self.group_key, self.json_key


@dataclass(frozen=True, slots=True)
class ConfigHealthIssue:
    source: str
    code: str
    message: str
    recoverable: bool = False


@dataclass(frozen=True, slots=True)
class ConfigHealth:
    issues: tuple[ConfigHealthIssue, ...] = ()

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)

    def format_lines(self) -> list[str]:
        if not self.issues:
            return []
        return [f"[{issue.source}] {issue.message}" for issue in self.issues]


def _setting(
    field_id: str,
    category: str,
    json_key: str,
    default: Any,
    value_type: type[Any],
    *,
    group_key: str | None = None,
    options: tuple[Any, ...] | None = None,
    label: str | None = None,
    description: str | None = None,
    visible: bool = True,
    coerce: Callable[[Any], Any] | None = None,
) -> SettingFieldSpec:
    return SettingFieldSpec(
        field_id=field_id,
        category=category,
        group_key=group_key,
        json_key=json_key,
        default=default,
        value_type=value_type,
        options=options,
        label=label,
        description=description,
        visible=visible,
        coerce=coerce,
    )


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise TypeError('value must be a list')
    return [item for item in value if isinstance(item, str)]


def _build_general_settings() -> tuple[SettingFieldSpec, ...]:
    return (
        _setting(
            "language",
            "General",
            "language",
            "English",
            str,
            options=("English", "French", "Spanish"),
        ),
        _setting(
            "launch_maximized",
            "General",
            "launch_maximized",
            True,
            bool,
            group_key="Display",
            options=(True, False),
        ),
        _setting(
            "theme",
            "General",
            "theme",
            "Auto",
            str,
            group_key="Display",
            options=("Light", "Dark", "Auto"),
        ),
    )


def _build_network_settings() -> tuple[SettingFieldSpec, ...]:
    return (
        _setting(
            "serial_baud_rate",
            "Network",
            "baud_rate",
            9600,
            int,
            group_key="Serial",
            options=(800, 1200, 2400, 4800, 9600, 14400, 19200, 38400),
        ),
        _setting(
            "serial_data_bits",
            "Network",
            "data_bits",
            8,
            int,
            group_key="Serial",
            options=(4, 8),
        ),
        _setting(
            "serial_stop_bits",
            "Network",
            "stop_bits",
            2,
            int,
            group_key="Serial",
            options=(1, 2, 4),
        ),
        _setting(
            "serial_parity",
            "Network",
            "parity",
            "None",
            str,
            group_key="Serial",
            options=("None", "Even", "Odd"),
        ),
        _setting(
            "use_proxy",
            "Network",
            "use_proxy",
            True,
            bool,
            group_key="Internet",
            options=(True, False),
        ),
        _setting(
            "proxy_address",
            "Network",
            "proxy_address",
            "127.0.0.1",
            str,
            group_key="Internet",
        ),
        _setting(
            "proxy_port",
            "Network",
            "proxy_port",
            "8080",
            str,
            group_key="Internet",
        ),
    )


def _build_task_center_settings() -> tuple[SettingFieldSpec, ...]:
    default_task_order = [spec.key for spec in get_task_specs()]
    return (
        _setting(
            "task_order",
            "Task_Center",
            "task_order",
            default_task_order,
            list,
            visible=False,
            coerce=_normalize_string_list,
        ),
    )


def _build_task_field(task_key: str, task_setting: TaskSettingSpec) -> SettingFieldSpec:
    return SettingFieldSpec(
        field_id=task_setting.setting_id,
        category="Batch_Files",
        group_key=task_key,
        json_key=task_setting.json_key,
        default=task_setting.default,
        value_type=task_setting.value_type,
        options=task_setting.options,
        label=task_setting.label,
        description=task_setting.description,
        visible=task_setting.visible,
        coerce=task_setting.coerce,
    )


def _build_task_settings() -> tuple[SettingFieldSpec, ...]:
    fields: list[SettingFieldSpec] = []
    for spec in get_task_specs():
        for task_setting in spec.settings:
            fields.append(_build_task_field(spec.key, task_setting))
    return tuple(fields)


def get_setting_field_specs() -> tuple[SettingFieldSpec, ...]:
    return (
        *_build_general_settings(),
        *_build_network_settings(),
        *_build_task_center_settings(),
        *_build_task_settings(),
    )


def get_setting_field_spec_map() -> dict[str, SettingFieldSpec]:
    return {spec.field_id: spec for spec in get_setting_field_specs()}


def build_default_settings_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for spec in get_setting_field_specs():
        cursor = payload
        path = spec.path
        for key in path[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[path[-1]] = deepcopy(spec.default)
    return payload
