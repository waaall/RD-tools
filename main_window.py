from __future__ import annotations

from PySide6.QtCore import QSize, QTimer
from PySide6.QtGui import QCloseEvent, QCursor, QGuiApplication, QScreen
from PySide6.QtWidgets import QApplication
from qfluentwidgets import (
    FluentIcon as FIF,
    FluentWindow,
    InfoBar,
    NavigationItemPosition,
)

from modules.app_settings import AppSettings
from ui.theme import (
    SYSTEM_THEME_SYNC_INTERVAL_MS,
    apply_app_theme,
    detect_system_theme,
    is_auto_theme,
    refresh_auto_app_theme,
)
from ui.task_descriptor import TaskDescriptor
from widgets.file_page import FileWindow
from widgets.help_page import HelpWindow
from widgets.setting_page import SettingWindow


class MainWindow(FluentWindow):
    AUTO_WINDOW_WIDTH_RATIO = 0.9
    AUTO_WINDOW_HEIGHT_RATIO = 0.9
    BASE_MINIMUM_SIZE = QSize(1180, 720)
    FALLBACK_WINDOW_SIZE = QSize(1320, 840)

    def __init__(self, settings: AppSettings, task_descriptors: list[TaskDescriptor] | None = None):
        super().__init__()
        self.settings = settings
        self.task_descriptors = list(task_descriptors or [])
        self._theme_sync_timer = QTimer(self)
        self._theme_sync_timer.setInterval(SYSTEM_THEME_SYNC_INTERVAL_MS)
        self._theme_sync_timer.timeout.connect(self._sync_auto_theme)
        self._last_system_theme = None

        self.setWindowTitle('RD-tools')
        self.resize(self.FALLBACK_WINDOW_SIZE)
        self.setMinimumSize(self.BASE_MINIMUM_SIZE)

        self._init_interfaces(self.task_descriptors)
        self._connect_signals()
        self._configure_theme_sync(self.settings.theme)
        self.switchTo(self.FileWindow)

    def _init_interfaces(self, task_descriptors: list[TaskDescriptor]):
        self.FileWindow = FileWindow()
        self.FileWindow.setObjectName('task-center')

        self.SettingWindow = SettingWindow(self.settings, task_descriptors)
        self.SettingWindow.setObjectName('settings-center')

        self.HelpWindow = HelpWindow()
        self.HelpWindow.setObjectName('help-center')

        self.addSubInterface(self.FileWindow, FIF.IOT, '任务中心', NavigationItemPosition.TOP)
        self.addSubInterface(self.SettingWindow, FIF.SETTING, '设置', NavigationItemPosition.TOP)
        self.addSubInterface(self.HelpWindow, FIF.HELP, '帮助', NavigationItemPosition.BOTTOM)

    def _connect_signals(self):
        self.FileWindow.notification_requested.connect(self.show_notification)
        self.FileWindow.task_order_changed.connect(self._handle_task_order_change)
        self.SettingWindow.notification_requested.connect(self.show_notification)
        self.SettingWindow.theme_changed.connect(self.apply_theme)
        self.SettingWindow.settings_reloaded.connect(self._handle_settings_reloaded)

    def show_notification(self, level: str, title: str, content: str, duration: int = 3000):
        notifier = getattr(InfoBar, level, InfoBar.info)
        notifier(title, content, duration=duration, parent=self)

    def apply_theme(self, theme_name: str):
        apply_app_theme(theme_name, QApplication.instance())
        self._configure_theme_sync(theme_name)
        self.FileWindow.refresh_log_view()

    def _configure_theme_sync(self, theme_name: str | None):
        if not is_auto_theme(theme_name):
            self._last_system_theme = None
            if self._theme_sync_timer.isActive():
                self._theme_sync_timer.stop()
            return

        self._last_system_theme = detect_system_theme()
        if not self._theme_sync_timer.isActive():
            self._theme_sync_timer.start()

    def _sync_auto_theme(self):
        if not is_auto_theme(self.settings.theme):
            self._configure_theme_sync(self.settings.theme)
            return

        current_system_theme = detect_system_theme()
        if current_system_theme == self._last_system_theme:
            return

        self._last_system_theme = current_system_theme
        refresh_auto_app_theme(QApplication.instance())
        self.FileWindow.refresh_log_view()

    def open_task_settings(self, task_key: str) -> bool:
        self.switchTo(self.SettingWindow)
        return self.SettingWindow.open_task_settings(task_key)

    def _handle_task_order_change(self, ordered_keys: list[str]):
        previous_order = [descriptor.key for descriptor in self.task_descriptors]
        if ordered_keys == previous_order:
            return

        descriptor_map = {descriptor.key: descriptor for descriptor in self.task_descriptors}
        reordered_descriptors = [
            descriptor_map[key]
            for key in ordered_keys
            if key in descriptor_map
        ]
        if len(reordered_descriptors) != len(self.task_descriptors):
            self.FileWindow.apply_task_order(previous_order)
            self.show_notification('error', '任务顺序', '任务顺序同步失败：当前任务列表不完整。')
            return

        if not self.settings.save_task_order(ordered_keys):
            # 配置写回失败时，界面必须和真实持久化状态保持一致，不能停留在“看起来成功”的顺序上。
            self.FileWindow.apply_task_order(previous_order)
            self.show_notification('error', '任务顺序', '任务顺序无法写回配置文件。')
            return

        self.task_descriptors = reordered_descriptors
        # 设置页不提供拖拽，但要跟随同一份有序 descriptor 立即刷新导航顺序。
        self.SettingWindow.set_task_descriptors(self.task_descriptors)

    def _handle_settings_reloaded(self):
        ordered_keys = self.settings.get_task_order([descriptor.key for descriptor in self.task_descriptors])
        descriptor_map = {descriptor.key: descriptor for descriptor in self.task_descriptors}
        self.task_descriptors = [
            descriptor_map[key]
            for key in ordered_keys
            if key in descriptor_map
        ]
        self.FileWindow.apply_task_order(ordered_keys)
        self.SettingWindow.set_task_descriptors(self.task_descriptors)

    def show_for_launch(self):
        self._apply_launch_geometry()
        if bool(self.settings.launch_maximized):
            self.showMaximized()
            return
        self.show()

    def _apply_launch_geometry(self):
        screen = self._get_launch_screen()
        if screen is None:
            self.resize(self.FALLBACK_WINDOW_SIZE)
            return

        available = screen.availableGeometry()
        minimum_width = min(self.BASE_MINIMUM_SIZE.width(), available.width())
        minimum_height = min(self.BASE_MINIMUM_SIZE.height(), available.height())
        self.setMinimumSize(minimum_width, minimum_height)

        width = min(
            max(int(available.width() * self.AUTO_WINDOW_WIDTH_RATIO), minimum_width),
            available.width(),
        )
        height = min(
            max(int(available.height() * self.AUTO_WINDOW_HEIGHT_RATIO), minimum_height),
            available.height(),
        )
        x = available.x() + max(0, (available.width() - width) // 2)
        y = available.y() + max(0, (available.height() - height) // 2)
        self.setGeometry(x, y, width, height)

    @staticmethod
    def _get_launch_screen() -> QScreen | None:
        return QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()

    def closeEvent(self, event: QCloseEvent):
        if self._theme_sync_timer.isActive():
            self._theme_sync_timer.stop()
        super().closeEvent(event)


if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    import sys

    from ui.task_ui_registry import build_task_descriptors

    app = QApplication(sys.argv)
    settings = AppSettings()
    apply_app_theme(settings.theme, app)
    trial = MainWindow(settings, build_task_descriptors())
    trial.show_for_launch()
    sys.exit(app.exec())
