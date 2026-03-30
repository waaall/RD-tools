from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QListWidgetItem,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    IndeterminateProgressRing,
    LineEdit,
    ListWidget,
    PrimaryPushButton,
    PushButton,
    SegmentedWidget,
    StrongBodyLabel,
    SubtitleLabel,
    TextEdit,
    TitleLabel,
    isDarkTheme,
)

from core import MessageLevel, TaskMessage, ensure_task_message
from ui.task_descriptor import TaskDescriptor


class FileWindow(QWidget):
    notification_requested = Signal(str, str, str)

    def __init__(self):
        super().__init__()

        self._work_folder = ''
        self._work_folder_items: list[str] = []
        self._task_descriptors: dict[str, TaskDescriptor] = {}
        self._task_callbacks: dict[str, Callable[[], None]] = {}
        self._task_settings_callbacks: dict[str, Callable[[], None]] = {}
        self._operation_logs: dict[str, list[TaskMessage]] = {}
        self._task_states: dict[str, str] = {}
        self._task_items: dict[str, QListWidgetItem] = {}
        self._task_has_settings: dict[str, bool] = {}
        self._running_tasks: set[str] = set()
        self._current_task_key = ''

        self.setObjectName('AppPage')
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(16)

        eyebrow = CaptionLabel('TASK CENTER', self)
        eyebrow.setObjectName('PageEyebrow')
        main_layout.addWidget(eyebrow)

        title = TitleLabel('任务中心', self)
        title.setObjectName('PageTitle')
        main_layout.addWidget(title)

        description = BodyLabel('所有任务共享同一份工作目录和处理范围，页面只负责统一调度、状态反馈和日志查看。', self)
        description.setObjectName('PageDescription')
        description.setWordWrap(True)
        main_layout.addWidget(description)

        self.work_folder_summary_label = CaptionLabel('当前未选择工作目录。', self)
        self.work_folder_summary_label.setObjectName('SummaryLabel')
        self.work_folder_summary_label.setWordWrap(True)
        main_layout.addWidget(self.work_folder_summary_label)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(16)
        main_layout.addLayout(body_layout, stretch=1)

        self.task_list_card = CardWidget(self)
        self.task_list_card.setObjectName('TaskListCard')
        self.task_list_card.setMinimumWidth(280)
        self.task_list_card.setMaximumWidth(340)
        task_list_layout = QVBoxLayout(self.task_list_card)
        task_list_layout.setContentsMargins(16, 16, 16, 16)
        task_list_layout.setSpacing(12)

        task_list_title = SubtitleLabel('任务列表', self.task_list_card)
        task_list_layout.addWidget(task_list_title)

        task_list_hint = CaptionLabel('切换任务不会丢失各自日志，运行中的任务会在列表里标记。', self.task_list_card)
        task_list_hint.setObjectName('SectionHint')
        task_list_hint.setWordWrap(True)
        task_list_layout.addWidget(task_list_hint)

        self.task_list = ListWidget(self.task_list_card)
        self.task_list.setObjectName('TaskList')
        self.task_list.currentItemChanged.connect(self._on_task_changed)
        task_list_layout.addWidget(self.task_list, stretch=1)
        body_layout.addWidget(self.task_list_card)

        content_layout = QVBoxLayout()
        content_layout.setSpacing(16)
        body_layout.addLayout(content_layout, stretch=1)

        self.overview_card = CardWidget(self)
        self.overview_card.setObjectName('SurfaceCard')
        overview_layout = QVBoxLayout(self.overview_card)
        overview_layout.setContentsMargins(16, 16, 16, 16)
        overview_layout.setSpacing(12)

        self.task_title_label = SubtitleLabel('请选择任务', self.overview_card)
        overview_layout.addWidget(self.task_title_label)

        self.task_description_label = BodyLabel('左侧列表中选中一个任务后，这里会展示任务说明和执行状态。', self.overview_card)
        self.task_description_label.setWordWrap(True)
        overview_layout.addWidget(self.task_description_label)

        status_layout = QHBoxLayout()
        status_layout.setSpacing(12)
        self.running_ring = IndeterminateProgressRing(self.overview_card, start=False)
        self.running_ring.setFixedSize(18, 18)
        self.running_ring.hide()
        status_layout.addWidget(self.running_ring, 0, Qt.AlignVCenter)

        self.task_state_label = StrongBodyLabel('待执行', self.overview_card)
        self.task_state_label.setObjectName('TaskStateLabel')
        status_layout.addWidget(self.task_state_label, 0, Qt.AlignVCenter)
        status_layout.addStretch(1)

        self.edit_settings_button = PushButton(self.overview_card)
        self.edit_settings_button.setObjectName('TaskSettingsButton')
        self.edit_settings_button.setText('修改设置')
        self.edit_settings_button.clicked.connect(self._open_current_task_settings)
        status_layout.addWidget(self.edit_settings_button)

        self.run_button = PrimaryPushButton(self.overview_card)
        self.run_button.setText('执行任务')
        self.run_button.clicked.connect(self._run_current_task)
        status_layout.addWidget(self.run_button)

        self.clear_log_button = PushButton(self.overview_card)
        self.clear_log_button.setText('清空日志')
        self.clear_log_button.clicked.connect(self._clear_current_log)
        status_layout.addWidget(self.clear_log_button)

        overview_layout.addLayout(status_layout)
        content_layout.addWidget(self.overview_card)

        self.content_card = CardWidget(self)
        self.content_card.setObjectName('SurfaceCard')
        content_card_layout = QVBoxLayout(self.content_card)
        content_card_layout.setContentsMargins(16, 16, 16, 16)
        content_card_layout.setSpacing(16)

        self.segmented_widget = SegmentedWidget(self.content_card)
        self.segmented_widget.setObjectName('SegmentHost')
        content_card_layout.addWidget(self.segmented_widget, 0, Qt.AlignLeft)

        self.segment_stack = QStackedWidget(self.content_card)
        content_card_layout.addWidget(self.segment_stack, stretch=1)

        self.range_page = QWidget(self.content_card)
        self._build_range_page()
        self.segment_stack.addWidget(self.range_page)

        self.log_page = QWidget(self.content_card)
        self._build_log_page()
        self.segment_stack.addWidget(self.log_page)

        self.segmented_widget.addItem('scope', '处理范围', lambda: self._switch_detail_page(self.range_page))
        self.segmented_widget.addItem('log', '运行日志', lambda: self._switch_detail_page(self.log_page))
        self.segmented_widget.setCurrentItem('scope')
        self.segment_stack.setCurrentWidget(self.range_page)

        content_layout.addWidget(self.content_card, stretch=1)

    def _build_range_page(self):
        layout = QVBoxLayout(self.range_page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        hint = CaptionLabel('工作目录只保留一份；任务执行时会复用这里的子目录选择。', self.range_page)
        hint.setObjectName('SectionHint')
        hint.setWordWrap(True)
        layout.addWidget(hint)

        path_layout = QHBoxLayout()
        path_layout.setSpacing(12)
        self.choose_folder_button = PushButton(self.range_page)
        self.choose_folder_button.setText('选择工作目录')
        self.choose_folder_button.clicked.connect(self._choose_work_folder)
        path_layout.addWidget(self.choose_folder_button)

        self.work_folder_display = LineEdit(self.range_page)
        self.work_folder_display.setReadOnly(True)
        self.work_folder_display.setPlaceholderText('请选择需要批处理的根目录')
        path_layout.addWidget(self.work_folder_display, stretch=1)
        layout.addLayout(path_layout)

        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(12)
        self.select_all_button = PushButton(self.range_page)
        self.select_all_button.setText('全选')
        self.select_all_button.clicked.connect(self.select_all_directories)
        actions_layout.addWidget(self.select_all_button)

        self.clear_selection_button = PushButton(self.range_page)
        self.clear_selection_button.setText('清空选择')
        self.clear_selection_button.clicked.connect(self.clear_selected_directories)
        actions_layout.addWidget(self.clear_selection_button)

        actions_layout.addStretch(1)
        self.selection_summary_label = CaptionLabel('请选择工作目录', self.range_page)
        self.selection_summary_label.setObjectName('SelectionSummary')
        actions_layout.addWidget(self.selection_summary_label)
        layout.addLayout(actions_layout)

        self.folder_list = ListWidget(self.range_page)
        self.folder_list.setObjectName('FolderList')
        self.folder_list.itemChanged.connect(self._update_selection_summary)
        layout.addWidget(self.folder_list, stretch=1)

        self.selection_status_label = CaptionLabel('未选择工作目录', self.range_page)
        self.selection_status_label.setObjectName('SelectionStatus')
        self.selection_status_label.setWordWrap(True)
        layout.addWidget(self.selection_status_label)
        self._set_status_label_state('neutral')

    def _build_log_page(self):
        layout = QVBoxLayout(self.log_page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        hint = CaptionLabel('过程消息只记录到当前任务日志，不再依赖底部状态栏。', self.log_page)
        hint.setObjectName('SectionHint')
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.log_view = TextEdit(self.log_page)
        self.log_view.setObjectName('TaskLogView')
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText('这里会显示当前任务的独立日志。')
        layout.addWidget(self.log_view, stretch=1)

    def register_task(
        self,
        descriptor: TaskDescriptor,
        run_callback: Callable[[], None],
        has_settings: bool = False,
        open_settings_callback: Callable[[], None] | None = None,
    ):
        self._task_descriptors[descriptor.key] = descriptor
        self._task_callbacks[descriptor.key] = run_callback
        self._task_has_settings[descriptor.key] = has_settings
        if open_settings_callback is not None:
            self._task_settings_callbacks[descriptor.key] = open_settings_callback
        self._operation_logs.setdefault(descriptor.key, [])
        self._task_states.setdefault(descriptor.key, '待执行')

        item = QListWidgetItem(descriptor.title)
        item.setIcon(descriptor.icon.qicon())
        item.setData(Qt.UserRole, descriptor.key)
        self.task_list.addItem(item)
        self._task_items[descriptor.key] = item

        if not self._current_task_key:
            self.task_list.setCurrentItem(item)

    def _on_task_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None):
        if not current:
            return

        self._current_task_key = current.data(Qt.UserRole)
        self._refresh_task_detail()

    def _refresh_task_detail(self):
        descriptor = self.current_task_descriptor()
        if descriptor is None:
            self.task_title_label.setText('请选择任务')
            self.task_description_label.setText('左侧列表中选中一个任务后，这里会展示任务说明和执行状态。')
            self.task_state_label.setText('待执行')
            self.running_ring.stop()
            self.running_ring.hide()
            self.run_button.setEnabled(False)
            self.edit_settings_button.setEnabled(False)
            self.log_view.clear()
            return

        self.task_title_label.setText(descriptor.title)
        self.task_description_label.setText(descriptor.description)
        self.task_state_label.setText(self._task_states.get(descriptor.key, '待执行'))
        is_running = descriptor.key in self._running_tasks
        self.running_ring.setVisible(is_running)
        if is_running:
            self.running_ring.start()
        else:
            self.running_ring.stop()
        self.run_button.setEnabled(not is_running)
        self._update_settings_button_state(descriptor.key)

        self._render_log_messages(self._operation_logs.get(descriptor.key, []))

    def _switch_detail_page(self, page: QWidget):
        self.segment_stack.setCurrentWidget(page)

    def _run_current_task(self):
        if not self._current_task_key:
            return
        callback = self._task_callbacks.get(self._current_task_key)
        if callback is not None:
            callback()

    def _clear_current_log(self):
        self.clear_operation_log(self._current_task_key)

    def _open_current_task_settings(self):
        if not self._current_task_key:
            return
        if not self._task_has_settings.get(self._current_task_key, False):
            return
        callback = self._task_settings_callbacks.get(self._current_task_key)
        if callback is not None:
            callback()

    def current_task_descriptor(self) -> TaskDescriptor | None:
        if not self._current_task_key:
            return None
        return self._task_descriptors.get(self._current_task_key)

    def append_operation_log(self, task_key: str, message: TaskMessage | str):
        resolved_message = ensure_task_message(message)
        if task_key not in self._operation_logs:
            self._operation_logs[task_key] = []

        self._operation_logs[task_key].append(resolved_message)
        if task_key == self._current_task_key:
            self._append_log_message_to_view(resolved_message)

    def clear_operation_log(self, task_key: str | None):
        if not task_key:
            return

        self._operation_logs[task_key] = []
        if task_key == self._current_task_key:
            self.log_view.clear()

    def log_operation_start(self, task_key: str, work_folder: str, wanted_items: list[str]):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.append_operation_log(task_key, TaskMessage.build(f'[{timestamp}] 开始执行'))
        self.append_operation_log(task_key, TaskMessage.build(f'工作目录: {work_folder}'))
        self.append_operation_log(task_key, TaskMessage.build(f"目标目录: {', '.join(wanted_items)}"))

    def get_selected_directories(self):
        selected_dirs = []
        for index in range(self.folder_list.count()):
            item = self.folder_list.item(index)
            if item.checkState() == Qt.Checked:
                selected_dirs.append(item.data(Qt.UserRole))
        return self._work_folder, selected_dirs

    def set_selection_status(self, message: str, is_error: bool = False):
        self.selection_status_label.setText(message)
        self._set_status_label_state('error' if is_error else 'success')

    def notify_blocking_issue(self, message: str):
        descriptor = self.current_task_descriptor()
        title = descriptor.title if descriptor else '任务中心'
        self.notification_requested.emit('error', title, message)

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
            self.selection_summary_label.setText('当前目录下没有可选子目录')
            self._refresh_work_folder_summary()
            return

        selected_count = len(self.get_selected_directories()[1])
        self.selection_summary_label.setText(f'已勾选 {selected_count} / {total_count} 个目录')
        self._refresh_work_folder_summary()

    def _choose_work_folder(self):
        start_dir = self._work_folder or os.getcwd()
        selected_folder = QFileDialog.getExistingDirectory(self, '选择目录', start_dir)

        if not selected_folder:
            return

        self._work_folder = selected_folder
        self.work_folder_display.setText(self._work_folder)
        self._load_work_folder_items()

    def _load_work_folder_items(self):
        self.folder_list.clear()
        self._work_folder_items = []

        try:
            self._work_folder_items = sorted(
                [
                    item for item in os.listdir(self._work_folder)
                    if not item.startswith('.') and os.path.isdir(os.path.join(self._work_folder, item))
                ]
            )
        except OSError as exc:
            self.set_selection_status(f'读取目录失败: {exc}', is_error=True)
            self.notification_requested.emit('error', '工作目录', f'无法读取目录内容: {exc}')
            self._refresh_work_folder_summary()
            return

        for item_name in self._work_folder_items:
            item = QListWidgetItem(item_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, item_name)
            self.folder_list.addItem(item)

        self._update_selection_summary()
        if self._work_folder_items:
            self.set_selection_status(f'已加载 {len(self._work_folder_items)} 个可选目录。')
        else:
            self.set_selection_status('当前工作目录下没有可处理的一级子目录。', is_error=True)

        self._refresh_work_folder_summary()

    def _refresh_work_folder_summary(self):
        if not self._work_folder:
            self.work_folder_summary_label.setText('当前未选择工作目录。')
            return

        total_count = self.folder_list.count()
        if total_count == 0:
            self.work_folder_summary_label.setText(f'工作目录：{self._work_folder} · 当前没有可处理的一级子目录。')
            return

        selected_count = len(self.get_selected_directories()[1])
        self.work_folder_summary_label.setText(
            f'工作目录：{self._work_folder} · 已勾选 {selected_count} / {total_count} 个子目录。'
        )

    def set_task_running(self, task_key: str, is_running: bool):
        if is_running:
            self._running_tasks.add(task_key)
            self._task_states[task_key] = '运行中'
            if task_key == self._current_task_key:
                self.running_ring.start()
        else:
            self._running_tasks.discard(task_key)
            if task_key == self._current_task_key:
                self.running_ring.stop()
        self._update_task_item(task_key)
        if task_key == self._current_task_key:
            self._refresh_task_detail()

    def finish_task(self, task_key: str, success: bool, message: str):
        descriptor = self._task_descriptors.get(task_key)
        task_title = descriptor.title if descriptor else '任务'
        self._task_states[task_key] = '上次运行完成' if success else '上次运行失败'
        self._update_task_item(task_key)

        if task_key == self._current_task_key:
            self._refresh_task_detail()

        if success:
            self.notification_requested.emit('success', task_title, message)
        else:
            self.notification_requested.emit('error', task_title, message)

    def _update_task_item(self, task_key: str):
        item = self._task_items.get(task_key)
        descriptor = self._task_descriptors.get(task_key)
        if item is None or descriptor is None:
            return

        title = descriptor.title
        if task_key in self._running_tasks:
            title = f'{title} · 运行中'
        item.setText(title)

    def _set_status_label_state(self, state: str):
        self.selection_status_label.setProperty('state', state)
        style = self.selection_status_label.style()
        style.unpolish(self.selection_status_label)
        style.polish(self.selection_status_label)
        self.selection_status_label.update()

    def _update_settings_button_state(self, task_key: str):
        has_settings = self._task_has_settings.get(task_key, False)
        self.edit_settings_button.setEnabled(True)
        self.edit_settings_button.setProperty('availability', 'enabled' if has_settings else 'disabled')
        self.edit_settings_button.setToolTip(
            '打开当前任务的设置项。' if has_settings else '当前任务没有可修改的设置选项。'
        )
        style = self.edit_settings_button.style()
        style.unpolish(self.edit_settings_button)
        style.polish(self.edit_settings_button)
        self.edit_settings_button.update()

    def refresh_log_view(self):
        if not self._current_task_key:
            return
        self._render_log_messages(self._operation_logs.get(self._current_task_key, []))

    def _render_log_messages(self, messages: list[TaskMessage]):
        self.log_view.clear()
        for message in messages:
            self._append_log_message_to_view(message)

    def _append_log_message_to_view(self, message: TaskMessage):
        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(message.text, self._log_char_format(message.level))
        cursor.insertBlock()
        self.log_view.setTextCursor(cursor)
        self.log_view.ensureCursorVisible()

    def _log_char_format(self, level: MessageLevel) -> QTextCharFormat:
        char_format = QTextCharFormat()
        color_map = self._log_color_map()
        color = color_map.get(level)
        if color is not None:
            char_format.setForeground(color)
        return char_format

    def _log_color_map(self) -> dict[MessageLevel, QColor | None]:
        if isDarkTheme():
            return {
                MessageLevel.INFO: None,
                MessageLevel.SUCCESS: QColor('#4ade80'),
                MessageLevel.WARNING: QColor('#fbbf24'),
                MessageLevel.ERROR: QColor('#f87171'),
            }

        return {
            MessageLevel.INFO: None,
            MessageLevel.SUCCESS: QColor('#15803d'),
            MessageLevel.WARNING: QColor('#b45309'),
            MessageLevel.ERROR: QColor('#dc2626'),
        }


# ===========================调试用==============================
def simple_main():
    app = QApplication(sys.argv)
    window = FileWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    simple_main()
