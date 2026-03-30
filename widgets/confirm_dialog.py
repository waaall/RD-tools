from __future__ import annotations

from PySide6.QtWidgets import QListWidget, QDialog, QHBoxLayout, QPlainTextEdit, QVBoxLayout
from qfluentwidgets import BodyLabel, PrimaryPushButton, PushButton, SubtitleLabel


class TaskExecutionConfirmDialog(QDialog):
    def __init__(
        self,
        task_title: str,
        selected_dirs: list[str],
        settings_lines: list[str] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("执行确认")
        self.resize(620, 520)
        self._build_ui(task_title, selected_dirs, settings_lines or [])

    def _build_ui(self, task_title: str, selected_dirs: list[str], settings_lines: list[str]):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = SubtitleLabel(f"执行任务: {task_title}", self)
        layout.addWidget(title)

        hint = BodyLabel("确认当前设置和目标目录后开始执行。", self)
        hint.setWordWrap(True)
        layout.addWidget(hint)

        if settings_lines:
            settings_title = SubtitleLabel("当前设置", self)
            layout.addWidget(settings_title)

            settings_view = QPlainTextEdit(self)
            settings_view.setReadOnly(True)
            settings_view.setPlainText("\n".join(settings_lines))
            settings_view.setMinimumHeight(180)
            layout.addWidget(settings_view)

        dirs_title = SubtitleLabel("勾选文件夹列表", self)
        layout.addWidget(dirs_title)

        dir_list = QListWidget(self)
        dir_list.addItems(selected_dirs)
        dir_list.setMinimumHeight(200)
        layout.addWidget(dir_list, stretch=1)

        button_layout = QHBoxLayout()
        button_layout.addStretch(1)

        cancel_button = PushButton("取消", self)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        confirm_button = PrimaryPushButton("确认执行", self)
        confirm_button.clicked.connect(self.accept)
        button_layout.addWidget(confirm_button)

        layout.addLayout(button_layout)

    @classmethod
    def confirm(
        cls,
        task_title: str,
        selected_dirs: list[str],
        settings_lines: list[str] | None = None,
        parent=None,
    ) -> bool:
        dialog = cls(task_title, selected_dirs, settings_lines=settings_lines, parent=parent)
        return dialog.exec() == QDialog.Accepted
