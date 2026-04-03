from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QListWidgetItem,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    ComboBox,
    FluentIcon as FIF,
    LineEdit,
    ListWidget,
    ScrollArea,
    SegmentedWidget,
    SettingCard,
    SettingCardGroup,
    SubtitleLabel,
    SwitchSettingCard,
    TitleLabel,
)

from modules.app_settings import AppSettings
from ui.task_descriptor import TaskDescriptor


def humanize_setting_label(value: str) -> str:
    parts = value.replace('-', '_').split('_')
    words = []
    for part in parts:
        if not part:
            continue
        words.append(part if part.isupper() else part.capitalize())
    return ' '.join(words)


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


@dataclass(frozen=True, slots=True)
class SettingsNavItem:
    key: str
    title: str
    icon: Any


class SettingsSplitView(QWidget):
    selection_changed = Signal(str)

    def __init__(self, title: str, hint: str, parent=None):
        super().__init__(parent)
        self._build_ui(title, hint)

    def _build_ui(self, title: str, hint: str):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        self.sidebar_card = CardWidget(self)
        self.sidebar_card.setObjectName('SettingsSidebarCard')
        self.sidebar_card.setMinimumWidth(280)
        self.sidebar_card.setMaximumWidth(340)
        sidebar_layout = QVBoxLayout(self.sidebar_card)
        sidebar_layout.setContentsMargins(16, 16, 16, 16)
        sidebar_layout.setSpacing(12)

        sidebar_title = SubtitleLabel(title, self.sidebar_card)
        sidebar_layout.addWidget(sidebar_title)

        sidebar_hint = CaptionLabel(hint, self.sidebar_card)
        sidebar_hint.setObjectName('SectionHint')
        sidebar_hint.setWordWrap(True)
        sidebar_layout.addWidget(sidebar_hint)

        self.nav_list = ListWidget(self.sidebar_card)
        self.nav_list.setObjectName('SettingsNavList')
        self.nav_list.currentItemChanged.connect(self._emit_selection)
        sidebar_layout.addWidget(self.nav_list, stretch=1)
        layout.addWidget(self.sidebar_card)

        self.detail_card = CardWidget(self)
        self.detail_card.setObjectName('SettingsDetailCard')
        detail_layout = QVBoxLayout(self.detail_card)
        detail_layout.setContentsMargins(16, 16, 16, 16)
        detail_layout.setSpacing(0)

        self.detail_scroll = ScrollArea(self.detail_card)
        self.detail_scroll.setWidgetResizable(True)
        self.detail_scroll.setFrameShape(QFrame.NoFrame)
        detail_layout.addWidget(self.detail_scroll, stretch=1)

        self.detail_container = QWidget(self.detail_scroll)
        self.detail_container.setObjectName('AppPage')
        self.detail_scroll.setWidget(self.detail_container)
        self.detail_scroll.setStyleSheet('QScrollArea { background: transparent; border: none; }')
        self.detail_scroll.viewport().setAutoFillBackground(False)
        self.detail_scroll.viewport().setStyleSheet('background: transparent;')

        self.detail_container_layout = QVBoxLayout(self.detail_container)
        self.detail_container_layout.setContentsMargins(0, 0, 0, 0)
        self.detail_container_layout.setSpacing(16)
        layout.addWidget(self.detail_card, stretch=1)

    def set_nav_items(self, items: list[SettingsNavItem]):
        self.nav_list.clear()
        for item_data in items:
            item = QListWidgetItem(item_data.title)
            icon = item_data.icon.qicon() if hasattr(item_data.icon, 'qicon') else item_data.icon
            item.setIcon(icon)
            item.setData(Qt.UserRole, item_data.key)
            self.nav_list.addItem(item)

    def ensure_selection(self):
        if self.nav_list.currentItem() is None and self.nav_list.count() > 0:
            self.nav_list.setCurrentRow(0)

    def clear_detail(self):
        while self.detail_container_layout.count():
            item = self.detail_container_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def show_empty_state(self, title: str, description: str):
        self.clear_detail()
        title_label = SubtitleLabel(title, self.detail_container)
        self.detail_container_layout.addWidget(title_label)

        description_label = BodyLabel(description, self.detail_container)
        description_label.setObjectName('PageDescription')
        description_label.setWordWrap(True)
        self.detail_container_layout.addWidget(description_label)
        self.detail_container_layout.addStretch(1)

    def _emit_selection(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None):
        if not current:
            return
        self.selection_changed.emit(current.data(Qt.UserRole))


class SettingWindow(QWidget):
    notification_requested = Signal(str, str, str)
    theme_changed = Signal(str)

    CATEGORY_ICONS = {
        'General': FIF.SETTING,
        'Network': FIF.IOT,
        'Batch_Files': FIF.DOCUMENT,
    }

    GENERAL_CATEGORIES = ('General', 'Network')

    def __init__(self, settings: AppSettings, task_descriptors: list[TaskDescriptor] | None = None):
        super().__init__()
        self.settings = settings
        self.task_descriptors = task_descriptors or []
        self.main_categories = self.settings.get_main_categories() or []
        self.general_categories = [name for name in self.main_categories if name in self.GENERAL_CATEGORIES]
        self._task_descriptor_map = {descriptor.key: descriptor for descriptor in self.task_descriptors}
        self._configured_task_groups = set(self.settings.get_setting_groups('Batch_Files'))

        self.setObjectName('AppPage')
        self._build_ui()
        self._populate_navigation()
        self._switch_panel(self.general_view, 'general')

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

        description = BodyLabel('设置项变更即时写回配置文件。通用设置与任务设置共享同一套侧栏 + 详情布局，不改变现有配置结构与任务分组命名。', self)
        description.setObjectName('PageDescription')
        description.setWordWrap(True)
        main_layout.addWidget(description)

        self.segmented_widget = SegmentedWidget(self)
        self.segmented_widget.setObjectName('SegmentHost')
        self.segmented_widget.addItem('general', '通用设置', lambda: self._switch_panel(self.general_view, 'general'))
        self.segmented_widget.addItem('tasks', '任务设置', lambda: self._switch_panel(self.task_view, 'tasks'))
        main_layout.addWidget(self.segmented_widget, 0, Qt.AlignLeft)

        self.panel_stack = QStackedWidget(self)
        self.general_view = SettingsSplitView('通用设置', 'General 与 Network 按分类呈现，Display 已并入 General 作为子分组。', self)
        self.task_view = SettingsSplitView('任务设置', '只显示存在独立设置项的任务，顺序、标题和图标与任务页保持一致。', self)
        self.panel_stack.addWidget(self.general_view)
        self.panel_stack.addWidget(self.task_view)
        main_layout.addWidget(self.panel_stack, stretch=1)

        self.general_view.selection_changed.connect(self._render_general_category)
        self.task_view.selection_changed.connect(self._render_task_settings)

    def _populate_navigation(self):
        general_items = [
            SettingsNavItem(key=category_name, title=category_name, icon=self.CATEGORY_ICONS.get(category_name, FIF.SETTING))
            for category_name in self.general_categories
        ]
        self.general_view.set_nav_items(general_items)

        task_items = []
        for descriptor in self.task_descriptors:
            if descriptor.settings_group not in self._configured_task_groups:
                continue
            task_items.append(SettingsNavItem(key=descriptor.key, title=descriptor.title, icon=descriptor.icon))
        self.task_view.set_nav_items(task_items)

    def set_task_descriptors(self, task_descriptors: list[TaskDescriptor]):
        current_task_key = None
        current_item = self.task_view.nav_list.currentItem()
        if current_item is not None:
            current_task_key = current_item.data(Qt.UserRole)

        self.task_descriptors = task_descriptors or []
        self._task_descriptor_map = {descriptor.key: descriptor for descriptor in self.task_descriptors}
        self._populate_navigation()

        if self.panel_stack.currentWidget() is self.task_view:
            if self.task_view.nav_list.count() == 0:
                self.task_view.show_empty_state('暂无任务设置', '当前没有可映射到 Batch_Files 的任务配置。')
                return

            # 任务中心改顺序后，设置页尽量保持用户当前查看的任务不变。
            if current_task_key is not None and self._select_nav_item(self.task_view, current_task_key):
                return
            self.task_view.ensure_selection()

    def _switch_panel(self, view: SettingsSplitView, key: str):
        self.panel_stack.setCurrentWidget(view)
        self.segmented_widget.setCurrentItem(key)
        if view.nav_list.count() == 0:
            if view is self.task_view:
                view.show_empty_state('暂无任务设置', '当前没有可映射到 Batch_Files 的任务配置。')
            else:
                view.show_empty_state('暂无通用设置', '当前未找到可显示的通用设置分类。')
            return
        view.ensure_selection()

    def _render_general_category(self, category_name: str):
        entries = self.settings.get_setting_entries(category_name)
        grouped_entries = self._group_entries(entries, start_index=1, default_group_title=category_name)
        icon = self.CATEGORY_ICONS.get(category_name, FIF.SETTING)
        self._render_grouped_entries(self.general_view, grouped_entries, icon)

    def _render_task_settings(self, task_key: str):
        descriptor = self._task_descriptor_map.get(task_key)
        if descriptor is None:
            self.task_view.show_empty_state('任务不存在', '未找到当前任务的元数据。')
            return

        entries = self.settings.get_setting_entries('Batch_Files', group_name=descriptor.settings_group)
        grouped_entries = self._group_entries(entries, start_index=2, default_group_title=descriptor.title)
        self._render_grouped_entries(self.task_view, grouped_entries, descriptor.icon)

    def _render_grouped_entries(self, view: SettingsSplitView, grouped_entries: list[tuple[str, list[dict[str, Any]]]], icon):
        view.clear_detail()
        if not grouped_entries:
            view.show_empty_state('暂无设置项', '当前选择下没有可展示的设置项。')
            return

        for group_title, entries in grouped_entries:
            group = SettingCardGroup(group_title, view.detail_container)
            for entry in entries:
                title = self._humanize(entry['path'][-1])
                card = self._build_setting_card(
                    name=entry['name'],
                    value=entry['value'],
                    options=entry['options'],
                    title=title,
                    content=None,
                    icon=icon,
                    parent=view.detail_container,
                )
                group.addSettingCard(card)
            view.detail_container_layout.addWidget(group)

        view.detail_container_layout.addStretch(1)

    def _group_entries(self, entries: list[dict[str, Any]], start_index: int, default_group_title: str):
        grouped_entries: dict[str, list[dict[str, Any]]] = {}
        default_title = self._humanize(default_group_title)
        for entry in entries:
            path = entry['path']
            group_title = ' / '.join(self._humanize(part) for part in path[start_index:-1]) or default_title
            grouped_entries.setdefault(group_title, []).append(entry)
        return list(grouped_entries.items())

    def has_task_settings(self, task_key: str) -> bool:
        descriptor = self._task_descriptor_map.get(task_key)
        if descriptor is None:
            return False
        return descriptor.settings_group in self._configured_task_groups

    def open_task_settings(self, task_key: str) -> bool:
        if not self.has_task_settings(task_key):
            return False

        self._switch_panel(self.task_view, 'tasks')
        return self._select_nav_item(self.task_view, task_key)

    def _build_setting_card(self, name: str, value: Any, options: list[Any] | None, title: str, content: str | None, icon, parent=None):
        if options is not None:
            if options and all(isinstance(option, bool) for option in options):
                card = SwitchSettingCard(icon, title, content, parent=parent or self)
                card.setChecked(bool(value))
                card.checkedChanged.connect(lambda checked, setting_name=name: self.update_setting(setting_name, checked))
                return card

            card = AppComboBoxSettingCard(icon, title, content, options=options, value=value, parent=parent or self)
            card.valueChanged.connect(lambda selected, setting_name=name: self.update_setting(setting_name, selected))
            return card

        is_password = 'api_key' in name.lower()
        text_value = '' if value is None else str(value)
        card = AppLineEditSettingCard(icon, title, content, value=text_value, is_password=is_password, parent=parent or self)
        card.valueChanged.connect(lambda text, setting_name=name: self.update_setting(setting_name, text))
        return card

    def update_setting(self, name: str, value):
        if not hasattr(self.settings, name):
            self.notification_requested.emit('error', '设置保存失败', f'未找到配置项: {name}')
            return

        # 真实状态以 AppSettings.save_settings() 的提交结果为准，避免先改内存、后写文件失败留下脏值。
        if not self.settings.save_settings(name, value):
            self.notification_requested.emit('error', '设置保存失败', f'{self._humanize(name)} 无法写回配置文件。')
            return

        if name == 'theme':
            self.theme_changed.emit(str(value))

    @staticmethod
    def _humanize(value: str) -> str:
        return humanize_setting_label(value)

    @staticmethod
    def _select_nav_item(view: SettingsSplitView, item_key: str) -> bool:
        for index in range(view.nav_list.count()):
            item = view.nav_list.item(index)
            if item.data(Qt.UserRole) != item_key:
                continue
            view.nav_list.setCurrentRow(index)
            return True
        return False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    trial = SettingWindow(AppSettings())
    trial.show()
    sys.exit(app.exec())
