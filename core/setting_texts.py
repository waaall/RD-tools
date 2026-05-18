"""设置页稳定 key 的显示文本翻译。

schema 里的 category/group/json_key 是持久化和绑定用的稳定标识；
本模块集中把这些 key 映射到英文源文本，并在 UI 渲染时按当前语言翻译。
"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QCoreApplication, QT_TRANSLATE_NOOP


SETTING_TEXT_CONTEXT = 'SettingsMeta'


def humanize_setting_key(value: str) -> str:
    """兜底把 snake/kebab key 转成英文标题，避免未登记 key 直接暴露下划线。"""
    if ' ' in value:
        return value
    parts = value.replace('-', '_').split('_')
    words = []
    for part in parts:
        if not part:
            continue
        words.append(part if part.isupper() else part.capitalize())
    return ' '.join(words)


# category / group / json_key / field_id 的统一显示源文本。
_SETTING_TEXT_SOURCES = {
    'General': QT_TRANSLATE_NOOP('SettingsMeta', 'General'),
    'Network': QT_TRANSLATE_NOOP('SettingsMeta', 'Network'),
    'Batch_Files': QT_TRANSLATE_NOOP('SettingsMeta', 'Task settings'),
    'Task_Center': QT_TRANSLATE_NOOP('SettingsMeta', 'Task Center'),
    'Display': QT_TRANSLATE_NOOP('SettingsMeta', 'Display'),
    'Serial': QT_TRANSLATE_NOOP('SettingsMeta', 'Serial'),
    'Internet': QT_TRANSLATE_NOOP('SettingsMeta', 'Internet'),
    'language': QT_TRANSLATE_NOOP('SettingsMeta', 'Language'),
    'launch_maximized': QT_TRANSLATE_NOOP('SettingsMeta', 'Launch maximized'),
    'theme': QT_TRANSLATE_NOOP('SettingsMeta', 'Theme'),
    'baud_rate': QT_TRANSLATE_NOOP('SettingsMeta', 'Baud rate'),
    'data_bits': QT_TRANSLATE_NOOP('SettingsMeta', 'Data bits'),
    'stop_bits': QT_TRANSLATE_NOOP('SettingsMeta', 'Stop bits'),
    'parity': QT_TRANSLATE_NOOP('SettingsMeta', 'Parity'),
    'use_proxy': QT_TRANSLATE_NOOP('SettingsMeta', 'Use proxy'),
    'proxy_address': QT_TRANSLATE_NOOP('SettingsMeta', 'Proxy address'),
    'proxy_port': QT_TRANSLATE_NOOP('SettingsMeta', 'Proxy port'),
    'task_order': QT_TRANSLATE_NOOP('SettingsMeta', 'Task order'),
    'log_folder_name': QT_TRANSLATE_NOOP('SettingsMeta', 'Log folder name'),
    'mode': QT_TRANSLATE_NOOP('SettingsMeta', 'Mode'),
    'pattern': QT_TRANSLATE_NOOP('SettingsMeta', 'Pattern'),
    'start_pattern': QT_TRANSLATE_NOOP('SettingsMeta', 'Start pattern'),
    'end_pattern': QT_TRANSLATE_NOOP('SettingsMeta', 'End pattern'),
    'replace_with': QT_TRANSLATE_NOOP('SettingsMeta', 'Replace with'),
    'include_extension': QT_TRANSLATE_NOOP('SettingsMeta', 'Include extension'),
    'case_sensitive': QT_TRANSLATE_NOOP('SettingsMeta', 'Case sensitive'),
    'recursive': QT_TRANSLATE_NOOP('SettingsMeta', 'Recursive'),
    'max_threads': QT_TRANSLATE_NOOP('SettingsMeta', 'Max threads'),
    'out_dir_prefix': QT_TRANSLATE_NOOP('SettingsMeta', 'Output directory prefix'),
    'AddGroupTitle': QT_TRANSLATE_NOOP('SettingsMeta', 'Add group title'),
    'GroupTitleMaxLength': QT_TRANSLATE_NOOP('SettingsMeta', 'Group title max length'),
    'model_path': QT_TRANSLATE_NOOP('SettingsMeta', 'Model path'),
    'parallel': QT_TRANSLATE_NOOP('SettingsMeta', 'Parallel processing'),
    'sampling_rate': QT_TRANSLATE_NOOP('SettingsMeta', 'Sampling rate'),
    'filter_low_cut': QT_TRANSLATE_NOOP('SettingsMeta', 'Filter low cut'),
    'filter_high_cut': QT_TRANSLATE_NOOP('SettingsMeta', 'Filter high cut'),
    'filter_order': QT_TRANSLATE_NOOP('SettingsMeta', 'Filter order'),
    'drop_raw_zero': QT_TRANSLATE_NOOP('SettingsMeta', 'Drop raw zero values'),
    'trim_raw_data': QT_TRANSLATE_NOOP('SettingsMeta', 'Trim raw data'),
    'trim_filtered_data': QT_TRANSLATE_NOOP('SettingsMeta', 'Trim filtered data'),
    'trim_percentage': QT_TRANSLATE_NOOP('SettingsMeta', 'Trim percentage'),
    'time_range_short': QT_TRANSLATE_NOOP('SettingsMeta', 'Short time range'),
    'time_range_medium': QT_TRANSLATE_NOOP('SettingsMeta', 'Medium time range'),
    'fps': QT_TRANSLATE_NOOP('SettingsMeta', 'FPS'),
    'frame_dpi': QT_TRANSLATE_NOOP('SettingsMeta', 'Frame DPI'),
}


_OPTION_TEXT_SOURCES = {
    ('theme', 'Light'): QT_TRANSLATE_NOOP('SettingsMeta', 'Light'),
    ('theme', 'Dark'): QT_TRANSLATE_NOOP('SettingsMeta', 'Dark'),
    ('theme', 'Auto'): QT_TRANSLATE_NOOP('SettingsMeta', 'Auto'),
    ('parity', 'None'): QT_TRANSLATE_NOOP('SettingsMeta', 'None'),
    ('parity', 'Even'): QT_TRANSLATE_NOOP('SettingsMeta', 'Even'),
    ('parity', 'Odd'): QT_TRANSLATE_NOOP('SettingsMeta', 'Odd'),
    ('mode', 'prefix'): QT_TRANSLATE_NOOP('SettingsMeta', 'Prefix'),
    ('mode', 'all'): QT_TRANSLATE_NOOP('SettingsMeta', 'All'),
    ('mode', 'body'): QT_TRANSLATE_NOOP('SettingsMeta', 'Body'),
    ('mode', 'between'): QT_TRANSLATE_NOOP('SettingsMeta', 'Between'),
}


def translate_setting_text(value: str | None) -> str:
    """翻译设置 category/group/key；未知 key 使用 humanize 兜底。"""
    if not value:
        return ''
    source = _SETTING_TEXT_SOURCES.get(value, humanize_setting_key(value))
    return QCoreApplication.translate(SETTING_TEXT_CONTEXT, source)


def translate_option_label(setting_key: str, value: Any) -> str:
    """翻译下拉选项显示文本，存储值保持不变。"""
    source = _OPTION_TEXT_SOURCES.get((setting_key, str(value)), str(value))
    return QCoreApplication.translate(SETTING_TEXT_CONTEXT, source)
