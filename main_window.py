from __future__ import annotations

from PySide6.QtWidgets import QApplication
from qfluentwidgets import FluentIcon as FIF, FluentWindow, InfoBar, NavigationItemPosition

from modules.app_settings import AppSettings
from ui.theme import apply_app_theme
from widgets.file_page import FileWindow
from widgets.help_page import HelpWindow
from widgets.setting_page import SettingWindow


class MainWindow(FluentWindow):
    def __init__(self, settings: AppSettings):
        super().__init__()
        self.settings = settings

        self.setWindowTitle('RD-tools')
        self.resize(1320, 840)
        self.setMinimumSize(1180, 720)

        self._init_interfaces()
        self._connect_signals()
        self.switchTo(self.FileWindow)

    def _init_interfaces(self):
        self.FileWindow = FileWindow()
        self.FileWindow.setObjectName('task-center')

        self.SettingWindow = SettingWindow(self.settings)
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

    def show_notification(self, level: str, title: str, content: str, duration: int = 3000):
        notifier = getattr(InfoBar, level, InfoBar.info)
        notifier(title, content, duration=duration, parent=self)

    def apply_theme(self, theme_name: str):
        apply_app_theme(theme_name, QApplication.instance())


if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    settings = AppSettings()
    apply_app_theme(settings.theme, app)
    trial = MainWindow(settings)
    trial.show()
    sys.exit(app.exec())
