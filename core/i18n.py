"""应用级国际化:语言模式、locale 解析、translator 安装。

英文是源语言(代码里的字符串字面量即英文),只有中文需要加载 .qm;
英文模式不安装任何 translator,缺失翻译时 Qt 天然回退到源串。
"""
from __future__ import annotations

from PySide6.QtCore import QLibraryInfo, QLocale, QTranslator

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

    system 跟随系统界面语言:系统界面是中文→zh_CN,其它一律回退英文。
    """
    if language_mode == LANGUAGE_ZH:
        return LANGUAGE_ZH
    if language_mode == LANGUAGE_EN:
        return LANGUAGE_EN
    if _system_prefers_chinese():
        return LANGUAGE_ZH
    return LANGUAGE_EN


def _system_prefers_chinese() -> bool:
    system_locale = QLocale.system()
    for language_tag in system_locale.uiLanguages():
        normalized_tag = language_tag.replace('_', '-').lower()
        if normalized_tag == 'zh' or normalized_tag.startswith('zh-'):
            return True
    return system_locale.language() == QLocale.Language.Chinese


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
