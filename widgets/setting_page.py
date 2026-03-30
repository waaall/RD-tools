from __future__ import annotations

import sys
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QApplication, QFrame, QLineEdit, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    ComboBox,
    LineEdit,
    ScrollArea,
    SettingCard,
    SettingCardGroup,
    StrongBodyLabel,
    SwitchSettingCard,
    TitleLabel,
    FluentIcon as FIF,
)

from modules.app_settings import AppSettings


class AppLineEditSettingCard(SettingCard):
    valueChanged = Signal(str)

    def __init__(self, icon, title: str, content: str | None = None, value: str = '', is_password: bool = False, parent=None):
        super().__init__(icon, title, content, parent)
        self.line_edit = LineEdit(self)
        self.line_edit.setText(value)
        self.line_edit.setClearButtonEnabled(True)
        self.line_edit.setFixedWidth(280)
        if is_password:
            self.line_edit.setEchoMode(QLineEdit.Password)
        self.hBoxLayout.addWidget(self.line_edit, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(16)
        self.line_edit.textChanged.connect(self.valueChanged.emit)

    def setValue(self, value: str):
        self.line_edit.setText(value)


class AppComboBoxSettingCard(SettingCard):
    valueChanged = Signal(object)

    def __init__(self, icon, title: str, content: str | None = None, options: list[Any] | None = None, value: Any = None, parent=None):
        super().__init__(icon, title, content, parent)
        self.combo_box = ComboBox(self)
        self.combo_box.setFixedWidth(220)
        self.hBoxLayout.addWidget(self.combo_box, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(16)
        self._options = options or []

        for option in self._options:
            self.combo_box.addItem(str(option), userData=option)

        if value in self._options:
            self.combo_box.setCurrentText(str(value))

        self.combo_box.currentIndexChanged.connect(self._emit_current_value)

    def _emit_current_value(self, index: int):
        self.valueChanged.emit(self.combo_box.itemData(index))

    def setValue(self, value: Any):
        self.combo_box.setCurrentText(str(value))


class SettingWindow(QWidget):
    notification_requested = Signal(str, str, str)
    theme_changed = Signal(str)

    CATEGORY_ICONS = {
        'General': FIF.SETTING,
        'Network': FIF.IOT,
        'Display': FIF.APPLICATION,
        'Batch_Files': FIF.DOCUMENT,
    }

    def __init__(self, settings: AppSettings):
        super().__init__()
        self.settings = settings
        self.main_categories = self.settings.get_main_categories() or []

        self.setObjectName('AppPage')
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(16)

        eyebrow = CaptionLabel('SETTINGS', self)
        eyebrow.setObjectName('PageEyebrow')
        main_layout.addWidget(eyebrow)

        title = TitleLabel('设置', self)
        title.setObjectName('PageTitle')
        main_layout.addWidget(title)

        description = BodyLabel('设置项变更即时写回配置文件。此页只重做表现层，不改变现有配置结构和信号链路。', self)
        description.setObjectName('PageDescription')
        description.setWordWrap(True)
        main_layout.addWidget(description)

        self.scroll_area = ScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        main_layout.addWidget(self.scroll_area, stretch=1)

        container = QWidget(self.scroll_area)
        container.setObjectName('AppPage')
        self.scroll_area.setWidget(container)

        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(16)

        for category_name in self.main_categories:
            group = SettingCardGroup(self._humanize(category_name), container)
            setting_map = self.settings.get_setting_map(category_name)
            for name, options_and_path in setting_map.items():
                options, path = self.settings._extract_options_path(options_and_path)
                value = getattr(self.settings, name, None)
                subgroup = ' / '.join(self._humanize(part) for part in path[1:-1]) or None
                title = self._humanize(path[-1])
                icon = self.CATEGORY_ICONS.get(category_name, FIF.SETTING)
                card = self._build_setting_card(name, value, options, title, subgroup, icon)
                group.addSettingCard(card)
            container_layout.addWidget(group)

        container_layout.addStretch(1)

    def _build_setting_card(self, name: str, value: Any, options: list[Any] | None, title: str, content: str | None, icon):
        if options is not None:
            if options and all(isinstance(option, bool) for option in options):
                card = SwitchSettingCard(icon, title, content, parent=self)
                card.setChecked(bool(value))
                card.checkedChanged.connect(lambda checked, setting_name=name: self.update_setting(setting_name, checked))
                return card

            card = AppComboBoxSettingCard(icon, title, content, options=options, value=value, parent=self)
            card.valueChanged.connect(lambda selected, setting_name=name: self.update_setting(setting_name, selected))
            return card

        is_password = 'api_key' in name.lower()
        text_value = '' if value is None else str(value)
        card = AppLineEditSettingCard(icon, title, content, value=text_value, is_password=is_password, parent=self)
        card.valueChanged.connect(lambda text, setting_name=name: self.update_setting(setting_name, text))
        return card

    def update_setting(self, name: str, value):
        if not hasattr(self.settings, name):
            self.notification_requested.emit('error', '设置保存失败', f'未找到配置项: {name}')
            return

        setattr(self.settings, name, value)
        if not self.settings.save_settings(name, value):
            self.notification_requested.emit('error', '设置保存失败', f'{self._humanize(name)} 无法写回配置文件。')
            return

        if name == 'theme':
            self.theme_changed.emit(str(value))

    @staticmethod
    def _humanize(value: str) -> str:
        parts = value.replace('-', '_').split('_')
        words = []
        for part in parts:
            if not part:
                continue
            words.append(part if part.isupper() else part.capitalize())
        return ' '.join(words)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    trial = SettingWindow(AppSettings())
    trial.show()
    sys.exit(app.exec())
