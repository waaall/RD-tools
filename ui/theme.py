from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication
from qfluentwidgets import StyleSheetBase, Theme, qconfig, setTheme, setThemeColor


APP_THEME_COLOR = '#2F6FED'


@dataclass(frozen=True, slots=True)
class AppThemeTokens:
    accent: str = '#2F6FED'
    success: str = '#2AA676'
    warning: str = '#D49C1D'
    error: str = '#C55252'
    dark_background: str = '#202124'
    dark_panel: str = '#26282C'
    dark_card: str = '#2C2F33'
    text_primary: str = '#F5F7FA'
    text_secondary: str = '#B8C0CC'


class AppStyleSheet(StyleSheetBase, Enum):
    APP = 'app'

    def path(self, theme: Theme = Theme.AUTO):
        active_theme = qconfig.theme if theme == Theme.AUTO else theme
        base_dir = Path(__file__).resolve().parent
        return str(base_dir / 'qss' / active_theme.value.lower() / f'{self.value}.qss')


TOKENS = AppThemeTokens()


def resolve_theme(theme_name: str | None) -> Theme:
    normalized_name = (theme_name or '').strip().lower()
    if normalized_name == 'light':
        return Theme.LIGHT
    if normalized_name == 'auto':
        return Theme.AUTO
    return Theme.DARK


def get_effective_theme(theme: Theme = Theme.AUTO) -> Theme:
    return qconfig.theme if theme == Theme.AUTO else theme


def load_app_stylesheet(app: QApplication | None = None, theme: Theme = Theme.AUTO) -> Theme:
    qt_app = app or QApplication.instance()
    if qt_app is None:
        raise RuntimeError('QApplication must exist before loading the app stylesheet.')

    effective_theme = get_effective_theme(theme)
    qss_path = AppStyleSheet.APP.path(effective_theme)
    with open(qss_path, 'r', encoding='utf-8') as f:
        qt_app.setStyleSheet(f.read())

    return effective_theme


def apply_app_theme(theme_name: str | None, app: QApplication | None = None) -> Theme:
    qt_app = app or QApplication.instance()
    if qt_app is None:
        raise RuntimeError('QApplication must exist before applying the app theme.')

    theme = resolve_theme(theme_name)
    setTheme(theme)
    setThemeColor(QColor(APP_THEME_COLOR))
    return load_app_stylesheet(qt_app, theme)
