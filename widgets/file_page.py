from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
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
    setCustomStyleSheet,
)

from core import MessageLevel, TaskMessage, ensure_task_message
from core.resource_paths import resolve_resource_path
from ui.task_descriptor import TaskDescriptor


class ReorderableTaskListWidget(ListWidget):
    order_changed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)

    def current_task_order(self):
        return [
            self.item(index).data(Qt.UserRole)
            for index in range(self.count())
        ]

    def dropEvent(self, event):
        previous_order = self.current_task_order()
        super().dropEvent(event)
        if not event.isAccepted():
            return

        current_order = self.current_task_order()
        # 只在顺序真的变化时通知上层，避免点击、取消拖拽也触发持久化。
        if current_order != previous_order:
            self.order_changed.emit(current_order)


class FileWindow(QWidget):
    notification_requested = Signal(str, str, str)
    task_order_changed = Signal(list)

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

        eyebrow = CaptionLabel(self.tr('TASK CENTER'), self)
        eyebrow.setObjectName('PageEyebrow')
        main_layout.addWidget(eyebrow)

        title = TitleLabel(self.tr('Task Center'), self)
        title.setObjectName('PageTitle')
        main_layout.addWidget(title)

        description = BodyLabel(self.tr('All tasks share one working directory and processing scope; this page only handles unified scheduling, status feedback, and log viewing.'), self)
        description.setObjectName('PageDescription')
        description.setWordWrap(True)
        main_layout.addWidget(description)

        self.work_folder_summary_label = CaptionLabel(self.tr('No working directory selected.'), self)
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

        task_list_title = SubtitleLabel(self.tr('Task list'), self.task_list_card)
        task_list_layout.addWidget(task_list_title)

        task_list_hint = CaptionLabel(self.tr('Drag items to reorder them; switching tasks keeps each log, and running tasks are marked in the list.'), self.task_list_card)
        task_list_hint.setObjectName('SectionHint')
        task_list_hint.setWordWrap(True)
        task_list_layout.addWidget(task_list_hint)

        self.task_list = ReorderableTaskListWidget(self.task_list_card)
        self.task_list.setObjectName('TaskList')
        self.task_list.currentItemChanged.connect(self._on_task_changed)
        self.task_list.order_changed.connect(self._on_task_order_changed)
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

        self.task_title_label = SubtitleLabel(self.tr('Select a task'), self.overview_card)
        overview_layout.addWidget(self.task_title_label)

        self.task_description_label = BodyLabel(self.tr('Select a task from the list on the left to see its description and run status here.'), self.overview_card)
        self.task_description_label.setWordWrap(True)
        overview_layout.addWidget(self.task_description_label)

        status_layout = QHBoxLayout()
        status_layout.setSpacing(12)
        self.running_ring = IndeterminateProgressRing(self.overview_card, start=False)
        self.running_ring.setFixedSize(18, 18)
        self.running_ring.hide()
        status_layout.addWidget(self.running_ring, 0, Qt.AlignVCenter)

        self.task_state_label = StrongBodyLabel(self.tr('Pending'), self.overview_card)
        self.task_state_label.setObjectName('TaskStateLabel')
        status_layout.addWidget(self.task_state_label, 0, Qt.AlignVCenter)
        status_layout.addStretch(1)

        self.edit_settings_button = PrimaryPushButton(self.overview_card)
        self.edit_settings_button.setObjectName('TaskSettingsButton')
        self.edit_settings_button.setText(self.tr('Edit settings'))
        self.edit_settings_button.clicked.connect(self._open_current_task_settings)
        self._apply_task_settings_button_theme()
        self.edit_settings_button.hide()
        status_layout.addWidget(self.edit_settings_button)

        self.run_button = PrimaryPushButton(self.overview_card)
        self.run_button.setText(self.tr('Run task'))
        self.run_button.clicked.connect(self._run_current_task)
        status_layout.addWidget(self.run_button)

        self.clear_log_button = PushButton(self.overview_card)
        self.clear_log_button.setText(self.tr('Clear log'))
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

        self.segmented_widget.addItem('scope', self.tr('Scope'), lambda: self._switch_detail_page(self.range_page))
        self.segmented_widget.addItem('log', self.tr('Log'), lambda: self._switch_detail_page(self.log_page))
        self.segmented_widget.setCurrentItem('scope')
        self.segment_stack.setCurrentWidget(self.range_page)

        content_layout.addWidget(self.content_card, stretch=1)

    def _build_range_page(self):
        layout = QVBoxLayout(self.range_page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        hint = CaptionLabel(self.tr('Only one working directory is kept; tasks reuse the subdirectory selection here when they run.'), self.range_page)
        hint.setObjectName('SectionHint')
        hint.setWordWrap(True)
        layout.addWidget(hint)

        path_layout = QHBoxLayout()
        path_layout.setSpacing(12)
        self.choose_folder_button = PushButton(self.range_page)
        self.choose_folder_button.setText(self.tr('Choose working directory'))
        self.choose_folder_button.clicked.connect(self._choose_work_folder)
        path_layout.addWidget(self.choose_folder_button)

        self.work_folder_display = LineEdit(self.range_page)
        self.work_folder_display.setReadOnly(True)
        self.work_folder_display.setPlaceholderText(self.tr('Select the root directory to batch-process'))
        path_layout.addWidget(self.work_folder_display, stretch=1)
        layout.addLayout(path_layout)

        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(12)
        self.select_all_button = PushButton(self.range_page)
        self.select_all_button.setText(self.tr('Select all'))
        self.select_all_button.clicked.connect(self.select_all_directories)
        actions_layout.addWidget(self.select_all_button)

        self.clear_selection_button = PushButton(self.range_page)
        self.clear_selection_button.setText(self.tr('Clear selection'))
        self.clear_selection_button.clicked.connect(self.clear_selected_directories)
        actions_layout.addWidget(self.clear_selection_button)

        actions_layout.addStretch(1)
        self.selection_summary_label = CaptionLabel(self.tr('Choose a working directory'), self.range_page)
        self.selection_summary_label.setObjectName('SelectionSummary')
        actions_layout.addWidget(self.selection_summary_label)
        layout.addLayout(actions_layout)

        self.folder_list = ListWidget(self.range_page)
        self.folder_list.setObjectName('FolderList')
        self.folder_list.itemChanged.connect(self._update_selection_summary)
        layout.addWidget(self.folder_list, stretch=1)

        self.selection_status_label = CaptionLabel(self.tr('No working directory selected'), self.range_page)
        self.selection_status_label.setObjectName('SelectionStatus')
        self.selection_status_label.setWordWrap(True)
        layout.addWidget(self.selection_status_label)
        self._set_status_label_state('neutral')

    def _build_log_page(self):
        layout = QVBoxLayout(self.log_page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        hint = CaptionLabel(self.tr('Process messages are recorded only in the current task log, not in the bottom status bar.'), self.log_page)
        hint.setObjectName('SectionHint')
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.log_view = TextEdit(self.log_page)
        self.log_view.setObjectName('TaskLogView')
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText(self.tr("The current task's log will appear here."))
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
        self._task_states.setdefault(descriptor.key, 'pending')

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

    def _on_task_order_changed(self, ordered_keys: list[str]):
        if ordered_keys:
            # 拖拽后尽量保持当前任务选中，避免右侧详情无意义地跳到别的任务。
            current_key = self._current_task_key or ordered_keys[0]
            self._select_task_item(current_key)
        self.task_order_changed.emit(ordered_keys)

    def _state_label(self, state_key: str) -> str:
        # 任务状态以稳定 key 存储,显示时按当前语言翻译
        labels = {
            'pending': self.tr('Pending'),
            'running': self.tr('Running'),
            'succeeded': self.tr('Last run succeeded'),
            'failed': self.tr('Last run failed'),
        }
        return labels.get(state_key, labels['pending'])

    def _refresh_task_detail(self):
        descriptor = self.current_task_descriptor()
        if descriptor is None:
            self.task_title_label.setText(self.tr('Select a task'))
            self.task_description_label.setText(self.tr('Select a task from the list on the left to see its description and run status here.'))
            self.task_state_label.setText(self._state_label('pending'))
            self.running_ring.stop()
            self.running_ring.hide()
            self.run_button.setEnabled(False)
            self.edit_settings_button.setEnabled(False)
            self.log_view.clear()
            return

        self.task_title_label.setText(descriptor.title)
        self.task_description_label.setText(descriptor.description)
        self.task_state_label.setText(self._state_label(self._task_states.get(descriptor.key, 'pending')))
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

    def current_task_order(self):
        return self.task_list.current_task_order()

    def apply_task_order(self, ordered_keys: list[str]):
        normalized_keys = [
            key for key in ordered_keys
            if key in self._task_items
        ]
        if not normalized_keys:
            return

        selected_key = self._current_task_key if self._current_task_key in normalized_keys else normalized_keys[0]
        self.task_list.setUpdatesEnabled(False)
        # 这里按目标行逐个搬移现有 item，而不是重建列表，避免打断现有选中态和运行状态标记。
        for target_row, key in enumerate(normalized_keys):
            item = self._task_items.get(key)
            if item is None:
                continue
            current_row = self.task_list.row(item)
            if current_row < 0 or current_row == target_row:
                continue
            moved_item = self.task_list.takeItem(current_row)
            self.task_list.insertItem(target_row, moved_item)
        self.task_list.setUpdatesEnabled(True)
        self._select_task_item(selected_key)

    def _select_task_item(self, task_key: str):
        item = self._task_items.get(task_key)
        if item is None:
            return
        self.task_list.setCurrentItem(item)

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
        self.append_operation_log(task_key, TaskMessage.build(self.tr('[{0}] Run started').format(timestamp)))
        self.append_operation_log(task_key, TaskMessage.build(self.tr('Working directory: {0}').format(work_folder)))
        self.append_operation_log(task_key, TaskMessage.build(self.tr('Target directories: {0}').format(', '.join(wanted_items))))

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
        title = descriptor.title if descriptor else self.tr('Task Center')
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
            self.selection_summary_label.setText(self.tr('No selectable subdirectories in the current directory'))
            self._refresh_work_folder_summary()
            return

        selected_count = len(self.get_selected_directories()[1])
        self.selection_summary_label.setText(self.tr('{0} / {1} directories selected').format(selected_count, total_count))
        self._refresh_work_folder_summary()

    def _choose_work_folder(self):
        start_dir = self._work_folder or os.getcwd()
        selected_folder = QFileDialog.getExistingDirectory(self, self.tr('Choose a directory'), start_dir)

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
            self.set_selection_status(self.tr('Failed to read directory: {0}').format(exc), is_error=True)
            self.notification_requested.emit('error', self.tr('Working directory'), self.tr('Unable to read directory contents: {0}').format(exc))
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
            self.set_selection_status(self.tr('Loaded {0} selectable directories.').format(len(self._work_folder_items)))
        else:
            self.set_selection_status(self.tr('The working directory has no first-level subdirectories to process.'), is_error=True)

        self._refresh_work_folder_summary()

    def _refresh_work_folder_summary(self):
        if not self._work_folder:
            self.work_folder_summary_label.setText(self.tr('No working directory selected.'))
            return

        total_count = self.folder_list.count()
        if total_count == 0:
            self.work_folder_summary_label.setText(
                self.tr('Working directory: {0} · no first-level subdirectories to process.').format(self._work_folder)
            )
            return

        selected_count = len(self.get_selected_directories()[1])
        self.work_folder_summary_label.setText(
            self.tr('Working directory: {0} · {1} / {2} subdirectories selected.').format(
                self._work_folder, selected_count, total_count)
        )

    def set_task_running(self, task_key: str, is_running: bool):
        if is_running:
            self._running_tasks.add(task_key)
            self._task_states[task_key] = 'running'
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
        task_title = descriptor.title if descriptor else self.tr('Task')
        self._task_states[task_key] = 'succeeded' if success else 'failed'
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
            title = self.tr('{0} · Running').format(title)
        item.setText(title)

    def _set_status_label_state(self, state: str):
        self.selection_status_label.setProperty('state', state)
        style = self.selection_status_label.style()
        style.unpolish(self.selection_status_label)
        style.polish(self.selection_status_label)
        self.selection_status_label.update()

    def _update_settings_button_state(self, task_key: str):
        has_settings = self._task_has_settings.get(task_key, False)
        self.edit_settings_button.setVisible(has_settings)
        self.edit_settings_button.setToolTip(self.tr('Open settings for this task.') if has_settings else '')

    def _apply_task_settings_button_theme(self):
        qss_root = resolve_resource_path('ui', 'qss')
        light_qss = (qss_root / 'light' / 'task_settings_button.qss').read_text(encoding='utf-8')
        dark_qss = (qss_root / 'dark' / 'task_settings_button.qss').read_text(encoding='utf-8')
        setCustomStyleSheet(self.edit_settings_button, light_qss, dark_qss)

    def refresh_log_view(self):
        if not self._current_task_key:
            return
        self._update_settings_button_state(self._current_task_key)
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
                MessageLevel.SUCCESS: QColor('#15803d'),
                MessageLevel.WARNING: QColor('#fbbf24'),
                MessageLevel.ERROR: QColor('#f87171'),
            }

        return {
            MessageLevel.INFO: None,
            MessageLevel.SUCCESS: QColor('#15803d'),
            MessageLevel.WARNING: QColor('#b45309'),
            MessageLevel.ERROR: QColor('#dc2626'),
        }


def simple_main():
    app = QApplication(sys.argv)
    window = FileWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    simple_main()
