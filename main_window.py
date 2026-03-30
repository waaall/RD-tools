import sys
from functools import partial

from PySide6.QtGui import *
from PySide6.QtCore import *
from PySide6.QtWidgets import *

from widgets import *


# =========================================================
# =======                 主界面类                 =========
# =========================================================
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        # 注意有一些初始化的顺序是不能更改的, 因为有依赖关系
        self.setWindowTitle("R&D_Tools")
        self.send_status_message("一切就绪, 确保您阅读文档, 再进行操作")

        # 初始化Dock和主窗口(centralWidget)
        self.__init_general_windows()
        self.setCentralWidget(self.Stack)

        # 初始化menu bar
        self.__createActions()
        self.__createMenu()

    # 初始化Dock和主窗口(centralWidget)
    def __init_general_windows(self):
        page_groups_names = ['帮助与设置', '批量处理']

        if hasattr(self, 'dock'):
            return
        self.dock = QDockWidget()
        self.dock.setWindowTitle('导航')
        self.dock_page = DockPage(page_groups_names)
        self.dock.setWidget(self.dock_page)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock)

        self.Stack = QStackedWidget()

        self.__help_window_name = 'HelpWindow'
        self.HelpWindow = HelpWindow()
        self.add_stack_page(self.HelpWindow, group_name=page_groups_names[0], button_name='帮助')

        self.SettingWindow = SettingWindow()
        self.add_stack_page(self.SettingWindow, group_name=page_groups_names[0], button_name='设置')

        self.FileWindow = FileWindow()
        self.add_stack_page(self.FileWindow, group_name=page_groups_names[1], button_name='任务')

    def add_stack_page(self, page_instance, group_name: str = '批量处理', button_name: str = None):
        """
        :param page_instance: 页面的类示例, 当然, 需要import你的页面类所在的文件
        :param group_name: dock_page的组名
        """
        page_name = page_instance.__class__.__name__
        button_name = button_name or page_name

        self.Stack.addWidget(page_instance)
        page_instance.setObjectName(page_name)

        self.dock_page.add_button(group_name, button_name,
                                  lambda: self.switch_stack_page(page_name))

        if hasattr(page_instance, 'result_signal'):
            page_instance.result_signal.connect(self.send_status_message)

    def show_dock(self):
        if hasattr(self, 'dock'):
            self.dock.show()
            self.statusBar().showMessage("导航已打开")
            return
        self.statusBar().showMessage("初始化导航")
        self.__init_general_windows()

    def send_status_message(self, message):
        self.statusBar().showMessage(message)

    def __createActions(self):
        # 创建 QAction 对象
        self.help_act = QAction("打开帮助", self, statusTip="帮助界面")
        # 连接 triggered 信号到槽函数，传递参数
        self.help_act.triggered.connect(partial(self.switch_stack_page, self.__help_window_name))
        self.userHelpAct = QAction("使用文档", statusTip="使用文档")

        self.userHelpAct.triggered.connect(self.show_user_help)
        self.devHelpAct = QAction("开发文档", statusTip="开发文档")
        self.devHelpAct.triggered.connect(self.show_dev_help)

        self.openDockAct = QAction("打开导航", statusTip="重新打开导航栏")
        self.openDockAct.triggered.connect(self.show_dock)

    def __createMenu(self):
        self.windowMenu = self.menuBar().addMenu("窗口")
        self.windowMenu.addAction(self.openDockAct)
        self.windowMenu.addAction(self.help_act)

        self.helpMenu = self.menuBar().addMenu("帮助")
        self.helpMenu.addAction(self.help_act)
        self.helpMenu.addAction(self.userHelpAct)
        self.helpMenu.addAction(self.devHelpAct)

    def show_user_help(self):
        self.switch_stack_page(self.__help_window_name)
        self.HelpWindow.show_user_manual()

    def show_dev_help(self):
        self.switch_stack_page(self.__help_window_name)
        self.HelpWindow.show_develop_manual()

    def switch_stack_page(self, window_name):
        widget = self.Stack.findChild(QWidget, window_name)
        if widget:
            self.Stack.setCurrentWidget(widget)
            self.fadeInWidget(self.Stack.currentWidget())
            self.statusBar().showMessage(f"open {window_name}")
        else:
            QMessageBox.about(self, 'error', f"Window:  '{window_name}' not found.")

    def fadeInWidget(self, new_widget):         # 有点假, 不需要old界面
        animationIn = QPropertyAnimation(new_widget)    # 动画的父控件为 login
        animationIn.setTargetObject(new_widget)         # 给register 做动画
        animationIn.setPropertyName(b"pos")
        animationIn.setStartValue(QPoint(-new_widget.width(), 0))
        animationIn.setEndValue(QPoint(0, 0))
        animationIn.setDuration(500)
        animationIn.setEasingCurve(QEasingCurve.InOutExpo)
        animationIn.start(QAbstractAnimation.DeleteWhenStopped)


# ===========================调试用==============================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    trial = MainWindow()
    trial.show()
    sys.exit(app.exec())
