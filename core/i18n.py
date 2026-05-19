"""应用级国际化:语言模式、locale 解析、translator 安装。

英文是源语言(代码里的字符串字面量即英文),只有中文需要加载 .qm;
英文模式不安装任何 translator,缺失翻译时 Qt 天然回退到源串。
"""
from __future__ import annotations

import sys

from PySide6.QtCore import QLibraryInfo, QLocale, QSettings, QTranslator

from core.resource_paths import resolve_resource_path

# 存进 settings.json 的稳定语言值
LANGUAGE_SYSTEM = 'system'
LANGUAGE_ZH = 'zh_CN'
LANGUAGE_EN = 'en'

# 语言下拉选项顺序:跟随系统 / 简体中文 / English
LANGUAGE_OPTIONS = (LANGUAGE_SYSTEM, LANGUAGE_ZH, LANGUAGE_EN)

# 旧版/非法 language 值 → 新稳定值,用于配置迁移
_LEGACY_LANGUAGE_ALIASES = {
    'system': LANGUAGE_SYSTEM,
    'english': LANGUAGE_EN,
    'en': LANGUAGE_EN,
    'chinese': LANGUAGE_ZH,
    'zh': LANGUAGE_ZH,
    'zh_cn': LANGUAGE_ZH,
    'zh-hans': LANGUAGE_ZH,
    '简体中文': LANGUAGE_ZH,
}

# 支持的 UI 语言映射:按 BCP-47 主语言码归一化,避免把某个地区/脚本写死。
_SUPPORTED_LOCALE_BY_LANGUAGE_CODE = {
    'zh': LANGUAGE_ZH,
    'en': LANGUAGE_EN,
}


def coerce_language(value: object) -> str:
    """把任意历史/非法 language 配置值规整成受支持的稳定值。

    旧 schema 用过 English/French/Spanish;无法识别的值一律回退“跟随系统”。
    """
    if isinstance(value, str):
        alias = _LEGACY_LANGUAGE_ALIASES.get(value.strip().lower())
        if alias is not None:
            return alias
    return LANGUAGE_SYSTEM


def resolve_locale(language_mode: str) -> str:
    """把 language 设置解析成实际生效的 locale(zh_CN 或 en)。

    system 跟随系统界面语言:按系统 UI 语言优先级选择第一个受支持语言。
    """
    if language_mode == LANGUAGE_ZH:
        return LANGUAGE_ZH
    if language_mode == LANGUAGE_EN:
        return LANGUAGE_EN
    return _resolve_system_locale()


def _resolve_system_locale() -> str:
    """解析系统 UI 语言,无法识别时回退英文源语言。"""
    system_locale = QLocale.system()
    language_tags = list(system_locale.uiLanguages())

    # macOS 下 python main.py 会继承 shell 的 LANG/LC_* 环境,QLocale.system()
    # 可能读到终端 locale,而不是“系统设置 > 语言与地区”的 UI 优先级。
    if sys.platform == 'darwin':
        macos_language_tags = _macos_ui_language_tags()
        if macos_language_tags:
            language_tags = macos_language_tags

    # 关键逻辑:按系统偏好顺序选第一个受支持语言,不能因为备用语言里有中文就直接切中文。
    for language_tag in language_tags:
        if not isinstance(language_tag, str):
            continue
        normalized_tag = language_tag.strip().replace('_', '-').lower()
        language_code = (
            normalized_tag
            .split('.', 1)[0]
            .split('@', 1)[0]
            .split('-', 1)[0]
        )
        locale = _SUPPORTED_LOCALE_BY_LANGUAGE_CODE.get(language_code)
        if locale is not None:
            return locale

    # 兜底:当系统只返回 C locale 等无法解析的标签时,再看 Qt 的 language 枚举。
    if system_locale.language() == QLocale.Language.Chinese:
        return LANGUAGE_ZH
    return LANGUAGE_EN


def _macos_ui_language_tags() -> list[str]:
    """读取 macOS 全局 UI 语言偏好列表 AppleLanguages。"""
    settings = QSettings(
        QSettings.Format.NativeFormat,
        QSettings.Scope.UserScope,
        'Apple',
        'Global Domain',
    )
    value = settings.value('AppleLanguages')
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [tag for tag in value if isinstance(tag, str)]
    return []


class TranslatorBundle:
    """持有当前安装的 QTranslator,切换语言时整体卸载再重装。

    包含三类:应用自身的 rdtools_*.qm、Qt 自带的 qtbase、qfluentwidgets 自带翻译。
    """

    def __init__(self) -> None:
        self._translators: list[QTranslator] = []

    def install(self, app, language_mode: str) -> str:
        """按语言模式安装 translator,返回实际生效的 locale。"""
        self.remove(app)
        locale = resolve_locale(language_mode)
        # 英文是源语言,不安装任何 translator
        if locale == LANGUAGE_EN:
            return locale
        self._install_app_translator(app, locale)
        self._install_qt_translator(app, locale)
        self._install_fluent_translator(app, locale)
        return locale

    def remove(self, app) -> None:
        """卸载并清空当前所有 translator。"""
        for translator in self._translators:
            app.removeTranslator(translator)
        self._translators.clear()

    def _install_app_translator(self, app, locale: str) -> None:
        try:
            qm_path = resolve_resource_path('i18n', f'rdtools_{locale}.qm')
        except FileNotFoundError:
            # .qm 尚未生成时静默跳过,UI 退回英文源串
            return
        translator = QTranslator(app)
        if translator.load(str(qm_path)):
            app.installTranslator(translator)
            self._translators.append(translator)

    def _install_qt_translator(self, app, locale: str) -> None:
        # Qt 自带控件(原生对话框、标准按钮等)的翻译
        translator = QTranslator(app)
        qt_dir = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
        if translator.load(f'qtbase_{locale}', qt_dir):
            app.installTranslator(translator)
            self._translators.append(translator)

    def _install_fluent_translator(self, app, locale: str) -> None:
        # qfluentwidgets 组件自带文案的翻译
        from qfluentwidgets import FluentTranslator

        translator = FluentTranslator(QLocale(locale), app)
        app.installTranslator(translator)
        self._translators.append(translator)
