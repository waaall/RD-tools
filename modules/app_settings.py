from __future__ import annotations

import json
import os
from copy import deepcopy

from PySide6.QtCore import QObject, Signal

from core.config_health import ConfigHealth, ConfigHealthIssue
from core.resource_paths import resolve_resource_path
from core.settings_schema import (
    SettingFieldSpec,
    build_default_settings_payload,
    get_setting_field_spec_map,
    get_setting_field_specs,
)
from core.task_registry import get_task_specs


class AppSettings(QObject):
    changed_signal = Signal(str, str, object)
    DEFAULT_TASK_ORDER = [spec.key for spec in get_task_specs()]

    def __init__(self):
        super().__init__()
        self._startup_warnings: list[str] = []
        self._config_issues: list[ConfigHealthIssue] = []
        self._field_specs: tuple[SettingFieldSpec, ...] = ()
        self._field_spec_by_id: dict[str, SettingFieldSpec] = {}
        self._main_categories: list[str] = []
        self.__settings_json: dict[str, object] = {}
        self.default_settings_file = self._resolve_default_settings_file()
        self.settings_file = self._resolve_user_settings_file()
        self.load_settings()

    def load_settings(self):
        self._field_specs = get_setting_field_specs()
        self._field_spec_by_id = get_setting_field_spec_map()
        self._config_issues = []

        default_payload = build_default_settings_payload()
        raw_snapshot = self._read_json_file(self.default_settings_file, source='仓库默认配置', recoverable=False)
        if raw_snapshot is not None and raw_snapshot != default_payload:
            self._record_config_issue(
                source='仓库默认配置',
                code='snapshot_mismatch',
                message='configs/settings.json 与 schema 默认配置不一致，运行时已使用 schema 默认值。',
                recoverable=False,
            )

        self._ensure_user_config_parent()
        if not os.path.exists(self.settings_file):
            self._write_settings_file(default_payload, target_path=self.settings_file)

        raw_user_settings = self._read_json_file(self.settings_file, source='用户配置', recoverable=True)
        effective_settings = deepcopy(default_payload)
        if raw_user_settings is not None:
            self._merge_user_settings(raw_user_settings, effective_settings)

        self.__settings_json = effective_settings
        self._main_categories = list(self.__settings_json.keys())
        self._sync_attributes_from_payload()
        return True

    def consume_startup_warnings(self):
        warnings = list(self._startup_warnings)
        self._startup_warnings.clear()
        return warnings

    def get_config_health(self) -> ConfigHealth:
        return ConfigHealth(tuple(self._config_issues))

    def get_main_categories(self):
        return list(self._main_categories)

    def get_task_order(self, available_task_keys: list[str]):
        available_keys = list(dict.fromkeys(
            key for key in available_task_keys
            if isinstance(key, str)
        ))
        available_key_set = set(available_keys)
        configured_order = getattr(self, 'task_order', []) or []

        ordered_keys = []
        seen = set()
        for key in configured_order:
            if key not in available_key_set or key in seen:
                continue
            seen.add(key)
            ordered_keys.append(key)

        for key in available_keys:
            if key in seen:
                continue
            seen.add(key)
            ordered_keys.append(key)

        return ordered_keys

    def save_task_order(self, task_keys: list[str]):
        normalized_keys = []
        seen = set()
        for key in task_keys:
            if not isinstance(key, str) or key in seen:
                continue
            seen.add(key)
            normalized_keys.append(key)
        return self.save_settings('task_order', normalized_keys)

    def get_setting_entries(self, category_name: str, group_name: str | None = None):
        entries = []
        for spec in self._field_specs:
            if not spec.visible or spec.category != category_name:
                continue
            if group_name is not None and spec.group_key != group_name:
                continue
            entries.append({
                'name': spec.field_id,
                'options': list(spec.options) if spec.options is not None else None,
                'path': spec.path,
                'value': getattr(self, spec.field_id, deepcopy(spec.default)),
                'label': spec.label,
                'description': spec.description,
            })
        return entries

    def get_setting_groups(self, category_name: str):
        groups = []
        seen = set()
        for entry in self.get_setting_entries(category_name):
            path = entry['path']
            if len(path) < 2:
                continue
            group_name = path[1]
            if group_name in seen:
                continue
            seen.add(group_name)
            groups.append(group_name)
        return groups

    def get_group_values(self, category_name: str, group_name: str) -> dict[str, object]:
        values: dict[str, object] = {}
        for spec in self._field_specs:
            if spec.category != category_name or spec.group_key != group_name:
                continue
            value = getattr(self, spec.field_id, deepcopy(spec.default))
            if value is None:
                continue
            values[spec.json_key] = value
        return values

    def get_value_from_path(self, path):
        value = self._lookup_path(self.__settings_json, path)
        return deepcopy(value)

    def save_settings(self, name: str, value):
        spec = self._field_spec_by_id.get(name)
        if spec is None:
            print(f"From AppSettings:\n\tSetting '{name}' not found\n")
            return False

        try:
            normalized_value = self._normalize_field_value(
                spec,
                value,
                source='用户配置',
                record_issue=False,
                record_warning=False,
            )
        except ValueError as exc:
            print(f"From AppSettings:\n\tSetting '{name}' invalid: {exc}\n")
            return False

        new_settings_json = deepcopy(self.__settings_json)
        self._set_value_at_path(new_settings_json, spec.path, normalized_value)

        if not self._write_settings_file(new_settings_json, target_path=self.settings_file):
            return False

        self.load_settings()
        return True

    def reset_user_settings_to_defaults(self):
        default_payload = build_default_settings_payload()
        if not self._write_settings_file(default_payload, target_path=self.settings_file):
            return False
        self.load_settings()
        return True

    def _sync_attributes_from_payload(self):
        for spec in self._field_specs:
            value = self.get_value_from_path(spec.path)
            setattr(self, spec.field_id, value)
            parent_key = spec.path[-2] if len(spec.path) >= 2 else spec.category
            self.changed_signal.emit(parent_key, spec.json_key, value)

    def _merge_user_settings(self, raw_user_settings: dict, effective_settings: dict):
        schema_tree = self._build_schema_tree()
        self._merge_schema_node(
            schema_node=schema_tree,
            raw_node=raw_user_settings,
            effective_node=effective_settings,
            current_path=(),
            source='用户配置',
        )

    def _merge_schema_node(self, schema_node: dict, raw_node, effective_node: dict, current_path: tuple[str, ...], source: str):
        if not isinstance(raw_node, dict):
            self._record_config_issue(
                source=source,
                code='invalid_container',
                message=f'{".".join(current_path) or "根节点"} 不是对象，已回退默认值。',
                recoverable=(source == '用户配置'),
            )
            return

        for key, raw_value in raw_node.items():
            if key not in schema_node:
                self._record_unknown_path(source, current_path + (key,))
                continue

            schema_child = schema_node[key]
            if isinstance(schema_child, SettingFieldSpec):
                try:
                    normalized_value = self._normalize_field_value(
                        schema_child,
                        raw_value,
                        source=source,
                    )
                except ValueError:
                    continue
                effective_node[key] = normalized_value
                continue

            if not isinstance(raw_value, dict):
                self._record_config_issue(
                    source=source,
                    code='invalid_container',
                    message=f'{".".join(current_path + (key,))} 应为对象，已回退默认值。',
                    recoverable=(source == '用户配置'),
                )
                continue

            self._merge_schema_node(
                schema_node=schema_child,
                raw_node=raw_value,
                effective_node=effective_node[key],
                current_path=current_path + (key,),
                source=source,
            )

    def _normalize_field_value(
        self,
        spec: SettingFieldSpec,
        raw_value,
        *,
        source: str,
        record_issue: bool = True,
        record_warning: bool = True,
    ):
        recoverable = source == '用户配置'
        path_text = '.'.join(spec.path)

        try:
            if spec.field_id == 'task_order':
                return self._normalize_task_order_value(raw_value, source=source, record_issue=record_issue, record_warning=record_warning)
            if spec.coerce is not None:
                value = spec.coerce(raw_value)
            elif spec.value_type is bool:
                value = self._coerce_bool(raw_value)
            elif spec.value_type is int:
                value = self._coerce_int(raw_value)
            elif spec.value_type is float:
                value = self._coerce_float(raw_value)
            elif spec.value_type is str:
                value = self._coerce_str(raw_value)
            elif spec.value_type is list:
                if not isinstance(raw_value, list):
                    raise ValueError('value must be a list')
                value = deepcopy(raw_value)
            else:
                if not isinstance(raw_value, spec.value_type):
                    raise ValueError(f'value must be {spec.value_type}')
                value = deepcopy(raw_value)
        except ValueError as exc:
            if record_issue:
                self._record_config_issue(
                    source=source,
                    code='invalid_value',
                    message=f'{path_text} 值非法({exc})，已回退默认值。',
                    recoverable=recoverable,
                )
            raise

        if spec.options is not None and value not in spec.options:
            if record_issue:
                self._record_config_issue(
                    source=source,
                    code='invalid_option',
                    message=f'{path_text} 不在允许选项中，已回退默认值。',
                    recoverable=recoverable,
                )
            raise ValueError('value is not in allowed options')

        return value

    def _normalize_task_order_value(self, raw_value, *, source: str, record_issue: bool, record_warning: bool):
        recoverable = source == '用户配置'
        default_task_order = [spec.key for spec in get_task_specs()]
        available_task_keys = set(default_task_order)

        if not isinstance(raw_value, list):
            if record_warning:
                self._record_startup_warning("Task_Center.task_order 配置非法，已回退到默认顺序。")
            if record_issue:
                self._record_config_issue(
                    source=source,
                    code='invalid_task_order',
                    message='Task_Center.task_order 不是列表，已回退默认顺序。',
                    recoverable=recoverable,
                )
            return list(default_task_order)

        source_order = []
        seen = set()
        has_non_string = False
        has_unknown = False
        has_duplicate = False

        for item in raw_value:
            if not isinstance(item, str):
                has_non_string = True
                continue
            if item in seen:
                has_duplicate = True
                continue
            seen.add(item)
            if item not in available_task_keys:
                has_unknown = True
                continue
            source_order.append(item)

        if has_non_string or has_unknown or has_duplicate:
            problems = []
            if has_non_string:
                problems.append("非字符串项")
            if has_unknown:
                problems.append("未知任务 key")
            if has_duplicate:
                problems.append("重复 key")
            if record_warning:
                self._record_startup_warning(
                    f"Task_Center.task_order 配置包含{' / '.join(problems)}，已过滤非法项并补齐缺失任务。"
                )
            if record_issue:
                self._record_config_issue(
                    source=source,
                    code='invalid_task_order_items',
                    message=f"Task_Center.task_order 包含{' / '.join(problems)}，已过滤并补齐默认顺序。",
                    recoverable=recoverable,
                )

        ordered_keys = []
        seen = set()
        for key in source_order:
            if key in seen:
                continue
            seen.add(key)
            ordered_keys.append(key)
        for key in default_task_order:
            if key in seen:
                continue
            seen.add(key)
            ordered_keys.append(key)
        return ordered_keys

    def _record_unknown_path(self, source: str, path: tuple[str, ...]):
        recoverable = source == '用户配置'
        path_text = '.'.join(path)
        if len(path) >= 2 and path[0] == 'Batch_Files' and path[1] not in {spec.key for spec in get_task_specs()}:
            message = f'{path_text} 不是已注册任务的设置分组，已忽略。'
        else:
            message = f'{path_text} 未在 schema 中定义，已忽略。'
        self._record_config_issue(
            source=source,
            code='unknown_path',
            message=message,
            recoverable=recoverable,
        )

    def _record_startup_warning(self, message: str):
        if message in self._startup_warnings:
            return
        print(f"From AppSettings:\n\tWarning: {message}\n")
        self._startup_warnings.append(message)

    def _record_config_issue(self, source: str, code: str, message: str, recoverable: bool):
        issue = ConfigHealthIssue(source=source, code=code, message=message, recoverable=recoverable)
        if issue in self._config_issues:
            return
        self._config_issues.append(issue)

    def _read_json_file(self, path: str, *, source: str, recoverable: bool):
        if not os.path.exists(path):
            self._record_config_issue(
                source=source,
                code='missing_file',
                message=f'{path} 不存在，已回退到 schema 默认值。',
                recoverable=recoverable,
            )
            return None

        try:
            with open(path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as exc:
            self._record_config_issue(
                source=source,
                code='parse_error',
                message=f'{path} 解析失败({exc})，已回退到 schema 默认值。',
                recoverable=recoverable,
            )
            return None

    def _resolve_default_settings_file(self) -> str:
        return str(resolve_resource_path('configs', 'settings.json'))

    def _resolve_user_settings_file(self) -> str:
        user_home = os.environ.get("HOME", "")
        if not user_home and os.name == 'nt':
            user_home = os.environ.get("USERPROFILE", "")
        user_config_dir = os.path.join(user_home, "Develop", "RD-tools-configs")
        return os.path.join(user_config_dir, "settings.json")

    def _ensure_user_config_parent(self):
        user_config_dir = os.path.dirname(self.settings_file)
        try:
            os.makedirs(user_config_dir, exist_ok=True)
        except Exception as exc:
            self._record_config_issue(
                source='用户配置',
                code='mkdir_failed',
                message=f'无法创建用户配置目录({exc})，将以内存默认值继续运行。',
                recoverable=True,
            )

    def _write_settings_file(self, settings_json: dict | None = None, *, target_path: str | None = None):
        payload = self.__settings_json if settings_json is None else settings_json
        path = target_path or self.settings_file
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as file:
                json.dump(payload, file, indent=4, ensure_ascii=False)
            return True
        except Exception as exc:
            print(f"From AppSettings:\n\t写入配置文件失败: {exc}\n")
            return False

    def _build_schema_tree(self):
        tree: dict[str, object] = {}
        for spec in self._field_specs:
            cursor = tree
            for key in spec.path[:-1]:
                cursor = cursor.setdefault(key, {})
            cursor[spec.path[-1]] = spec
        return tree

    @staticmethod
    def _lookup_path(source: dict | None, path):
        cursor = source
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                return None
            cursor = cursor[key]
        return cursor

    @staticmethod
    def _set_value_at_path(payload: dict, path: tuple[str, ...], value):
        cursor = payload
        for key in path[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[path[-1]] = deepcopy(value)

    @staticmethod
    def _coerce_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ('true', 'false'):
                return normalized == 'true'
        raise ValueError('value must be a boolean')

    @staticmethod
    def _coerce_int(value) -> int:
        if isinstance(value, bool):
            raise ValueError('bool is not allowed as int')
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith(('+', '-')):
                sign = stripped[0]
                digits = stripped[1:]
                if digits.isdigit():
                    return int(sign + digits)
            elif stripped.isdigit():
                return int(stripped)
        raise ValueError('value must be an integer')

    @staticmethod
    def _coerce_float(value) -> float:
        if isinstance(value, bool):
            raise ValueError('bool is not allowed as float')
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            try:
                return float(stripped)
            except ValueError as exc:
                raise ValueError('value must be a float') from exc
        raise ValueError('value must be a float')

    @staticmethod
    def _coerce_str(value) -> str:
        if value is None:
            raise ValueError('value must not be null')
        return str(value)
