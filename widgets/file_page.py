import os
import sys

from PySide6.QtGui import *
from PySide6.QtCore import *
from PySide6.QtWidgets import *


# =========================================================
# =======                文件操作界面               =========
# =========================================================
class FileWindow(QWidget):
    def __init__(self):
        super().__init__()

        self._work_folder = ""
        self._work_folder_items = []
        self._operation_logs = {}

        self.window_minimum_size = [260, 320, 220, 220]
        main_layout = QVBoxLayout()

        self.__init_workfolder_group()
        main_layout.addWidget(self.workfolder_group, stretch=2)

        self.__init_task_group()
        main_layout.addWidget(self.task_group, stretch=3)

        self.setWindowTitle("文件窗口")
        self.setLayout(main_layout)

    def __init_workfolder_group(self):
        self.workfolder_group = QGroupBox("一: 选择输入目录", self)
        group_layout = QVBoxLayout()

        path_layout = QHBoxLayout()
        choose_folder_button = QPushButton("选择工作目录", self)
        choose_folder_button.clicked.connect(self.__get_work_folder)
        path_layout.addWidget(choose_folder_button)

        self.work_folder_display = QLineEdit(self)
        self.work_folder_display.setPlaceholderText("请选择需要批处理的根目录")
        self.work_folder_display.setReadOnly(True)
        path_layout.addWidget(self.work_folder_display, stretch=1)
        group_layout.addLayout(path_layout)

        helper_label = QLabel("勾选当前工作目录下需要处理的一级子目录。每个任务会复用这份选择。", self)
        helper_label.setWordWrap(True)
        group_layout.addWidget(helper_label)

        self.folder_list = QListWidget(self)
        self.folder_list.setMinimumSize(self.window_minimum_size[0], self.window_minimum_size[2])
        self.folder_list.itemChanged.connect(self._update_selection_summary)
        group_layout.addWidget(self.folder_list)

        actions_layout = QHBoxLayout()
        select_all_button = QPushButton("全选", self)
        select_all_button.clicked.connect(self.select_all_directories)
        actions_layout.addWidget(select_all_button)

        clear_button = QPushButton("清空选择", self)
        clear_button.clicked.connect(self.clear_selected_directories)
        actions_layout.addWidget(clear_button)

        actions_layout.addStretch(1)

        self.selection_summary_label = QLabel("请选择工作目录", self)
        actions_layout.addWidget(self.selection_summary_label)
        group_layout.addLayout(actions_layout)

        self.selection_status_label = QLabel("未选择工作目录", self)
        self.selection_status_label.setWordWrap(True)
        group_layout.addWidget(self.selection_status_label)

        self.workfolder_group.setLayout(group_layout)

    def __init_task_group(self):
        self.task_group = QGroupBox("二: 任务卡片", self)
        group_layout = QVBoxLayout()

        helper_label = QLabel("每张卡片都会使用上面勾选的目录范围，并把日志保留在自己的卡片里。", self)
        helper_label.setWordWrap(True)
        group_layout.addWidget(helper_label)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        self.task_cards_container = QWidget(self)
        self.task_cards_layout = QVBoxLayout(self.task_cards_container)
        self.task_cards_layout.setAlignment(Qt.AlignTop)
        scroll_area.setWidget(self.task_cards_container)

        group_layout.addWidget(scroll_area)
        self.task_group.setLayout(group_layout)

    def add_file_operation(self, name, description, slot_func):
        card = QFrame(self.task_cards_container)
        card.setFrameShape(QFrame.StyledPanel)
        card_layout = QVBoxLayout(card)

        title_label = QLabel(name, card)
        title_font = title_label.font()
        title_font.setBold(True)
        title_font.setPointSize(title_font.pointSize() + 1)
        title_label.setFont(title_font)
        card_layout.addWidget(title_label)

        description_label = QLabel(description, card)
        description_label.setWordWrap(True)
        card_layout.addWidget(description_label)

        button_layout = QHBoxLayout()
        run_button = QPushButton("执行任务", card)
        run_button.clicked.connect(slot_func)
        button_layout.addWidget(run_button)

        clear_log_button = QPushButton("清空日志", card)
        clear_log_button.clicked.connect(lambda: self.clear_operation_log(name))
        button_layout.addWidget(clear_log_button)
        button_layout.addStretch(1)
        card_layout.addLayout(button_layout)

        result_display = QPlainTextEdit(card)
        result_display.setMinimumHeight(140)
        result_display.setReadOnly(True)
        result_display.setPlaceholderText("这里会显示该任务的独立日志。")
        card_layout.addWidget(result_display)

        self._operation_logs[name] = result_display
        self.task_cards_layout.addWidget(card)

    def append_operation_log(self, operation_name, message):
        result_display = self._operation_logs.get(operation_name)
        if result_display is None:
            print(f"Unknown operation: {operation_name}")
            return

        result_display.appendPlainText(message)
        result_display.moveCursor(QTextCursor.End)

    def clear_operation_log(self, operation_name):
        result_display = self._operation_logs.get(operation_name)
        if result_display is not None:
            result_display.clear()

    def log_operation_start(self, operation_name, work_folder, wanted_items):
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        self.append_operation_log(operation_name, f"[{timestamp}] 开始执行")
        self.append_operation_log(operation_name, f"工作目录: {work_folder}")
        self.append_operation_log(operation_name, f"目标目录: {', '.join(wanted_items)}")

    def get_selected_directories(self):
        selected_dirs = []
        for index in range(self.folder_list.count()):
            item = self.folder_list.item(index)
            if item.checkState() == Qt.Checked:
                selected_dirs.append(item.data(Qt.UserRole))
        return self._work_folder, selected_dirs

    def set_selection_status(self, message, is_error=False):
        self.selection_status_label.setText(message)
        color = "#b00020" if is_error else "#2e7d32"
        self.selection_status_label.setStyleSheet(f"color: {color};")

    def select_all_directories(self):
        for index in range(self.folder_list.count()):
            item = self.folder_list.item(index)
            item.setCheckState(Qt.Checked)

    def clear_selected_directories(self):
        for index in range(self.folder_list.count()):
            item = self.folder_list.item(index)
            item.setCheckState(Qt.Unchecked)

    def _update_selection_summary(self, _item=None):
        total_count = self.folder_list.count()
        if total_count == 0:
            self.selection_summary_label.setText("当前目录下没有可选子目录")
            return

        selected_count = len(self.get_selected_directories()[1])
        self.selection_summary_label.setText(f"已勾选 {selected_count} / {total_count} 个目录")

    def __get_work_folder(self):
        start_dir = self._work_folder if self._work_folder else "./"
        selected_folder = QFileDialog.getExistingDirectory(self, "选择目录", start_dir)

        if not selected_folder:
            print("From FileWindow:\n\tError: 未选择目录")
            return

        self._work_folder = selected_folder
        self.work_folder_display.setText(self._work_folder)
        self._load_work_folder_items()

    def _load_work_folder_items(self):
        self.folder_list.clear()
        self._work_folder_items = sorted(
            [
                item for item in os.listdir(self._work_folder)
                if not item.startswith('.')
                and os.path.isdir(os.path.join(self._work_folder, item))
            ]
        )

        for item_name in self._work_folder_items:
            item = QListWidgetItem(item_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, item_name)
            self.folder_list.addItem(item)

        self._update_selection_summary()

        if self._work_folder_items:
            self.set_selection_status(f"已加载 {len(self._work_folder_items)} 个可选目录。")
        else:
            self.set_selection_status("当前工作目录下没有可处理的一级子目录。", is_error=True)


# ===========================调试用==============================
def simple_main():
    app = QApplication(sys.argv)
    window = FileWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    simple_main()
