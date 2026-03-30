from __future__ import annotations

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication
from qfluentwidgets import (
    FluentIcon as FIF,
    FluentWindow,
    InfoBar,
    NavigationItemPosition,
    SystemThemeListener,
)

from modules.app_settings import AppSettings
from ui.theme import apply_app_theme
from ui.task_descriptor import TaskDescriptor
from widgets.file_page import FileWindow
from widgets.help_page import HelpWindow
from widgets.setting_page import SettingWindow


class MainWindow(FluentWindow):
    def __init__(self, settings: AppSettings, task_descriptors: list[TaskDescriptor] | None = None):
        super().__init__()
        self.settings = settings
        self._theme_listener = SystemThemeListener(self)

        self.setWindowTitle('RD-tools')
        self.resize(1320, 840)
        self.setMinimumSize(1180, 720)

        self._init_interfaces(task_descriptors or [])
        self._connect_signals()
        self._theme_listener.start()
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
        self.SettingWindow.notification_requested.connect(self.show_notification)
        self.SettingWindow.theme_changed.connect(self.apply_theme)
        self._theme_listener.systemThemeChanged.connect(self.refresh_theme)

    def show_notification(self, level: str, title: str, content: str, duration: int = 3000):
        notifier = getattr(InfoBar, level, InfoBar.info)
        notifier(title, content, duration=duration, parent=self)

    def apply_theme(self, theme_name: str):
        apply_app_theme(theme_name, QApplication.instance())
        self.FileWindow.refresh_log_view()

    def refresh_theme(self):
        apply_app_theme(self.settings.theme, QApplication.instance())
        self.FileWindow.refresh_log_view()

    def open_task_settings(self, task_key: str) -> bool:
        self.switchTo(self.SettingWindow)
        return self.SettingWindow.open_task_settings(task_key)

    def closeEvent(self, event: QCloseEvent):
        if self._theme_listener.isRunning():
            self._theme_listener.requestInterruption()
            self._theme_listener.wait(2000)
        super().closeEvent(event)


if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    settings = AppSettings()
    apply_app_theme(settings.theme, app)
    trial = MainWindow(settings)
    trial.show()
    sys.exit(app.exec())
