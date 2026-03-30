from __future__ import annotations

from PySide6.QtWidgets import QAbstractItemView, QApplication, QDialog
from qfluentwidgets import BodyLabel, ListWidget, MessageBoxBase, PlainTextEdit, SubtitleLabel


class TaskExecutionConfirmDialog(MessageBoxBase):
    def __init__(
        self,
        task_title: str,
        selected_dirs: list[str],
        settings_lines: list[str] | None = None,
        parent=None,
    ):
        dialog_parent = parent or QApplication.activeWindow()
        if dialog_parent is None:
            raise RuntimeError('TaskExecutionConfirmDialog requires a parent window.')

        super().__init__(parent=dialog_parent)

        self.yesButton.setText('确认执行')
        self.cancelButton.setText('取消')

        self._build_ui(task_title, selected_dirs, settings_lines or [])

    def _build_ui(self, task_title: str, selected_dirs: list[str], settings_lines: list[str]):
        title = SubtitleLabel(f'执行任务: {task_title}', self.widget)
        title.setObjectName('TaskConfirmTitle')
        self.viewLayout.addWidget(title)

        hint = BodyLabel('确认当前设置和目标目录后开始执行。', self.widget)
        hint.setObjectName('TaskConfirmHint')
        hint.setWordWrap(True)
        self.viewLayout.addWidget(hint)

        if settings_lines:
            settings_title = BodyLabel('当前设置', self.widget)
            settings_title.setObjectName('TaskConfirmSectionTitle')
            self.viewLayout.addWidget(settings_title)

            settings_view = PlainTextEdit(self.widget)
            settings_view.setObjectName('TaskConfirmSettingsView')
            settings_view.setReadOnly(True)
            settings_view.setPlainText('\n'.join(settings_lines))
            settings_view.setFixedHeight(180)
            self.viewLayout.addWidget(settings_view)

        dirs_title = BodyLabel('勾选文件夹列表', self.widget)
        dirs_title.setObjectName('TaskConfirmSectionTitle')
        self.viewLayout.addWidget(dirs_title)

        dir_list = ListWidget(self.widget)
        dir_list.setObjectName('TaskConfirmDirList')
        dir_list.setSelectionMode(QAbstractItemView.NoSelection)
        dir_list.addItems(selected_dirs)
        dir_list.setFixedHeight(220)
        self.viewLayout.addWidget(dir_list)

        self.widget.setFixedWidth(680)
        self.widget.layout().activate()
        self.widget.setFixedHeight(self.widget.sizeHint().height())

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
