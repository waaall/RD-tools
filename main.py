from __future__ import annotations

import sys

from PySide6.QtCore import QThread, Qt, QTimer, Signal
from PySide6.QtWidgets import QApplication

from core import MessageLevel, TaskLoader, TaskMessage
from core.task_params import build_task_params
from main_window import MainWindow
from modules.app_settings import AppSettings
from modules.files_basic import FilesBasic
from ui import TaskDescriptor, apply_app_theme
from ui.task_ui_registry import build_task_descriptors
from widgets.confirm_dialog import TaskExecutionConfirmDialog
from widgets.setting_page import humanize_setting_label


class BatchFilesBinding(QThread):
    result_signal = Signal(str, object)
    running_changed = Signal(str, bool)
    completed = Signal(str, bool, str)

    def __init__(self, descriptor: TaskDescriptor, settings: AppSettings, work_folder: str, wanted_items: list[str]):
        super().__init__()
        self.work_folder = work_folder
        self.wanted_items = wanted_items
        self.descriptor = descriptor
        self.task_key = descriptor.key
        self.display_name = descriptor.title
        self.settings = settings
        self._run_error: str | None = None

        self.finished.connect(self._handle_finished)

    def _forward_result(self, message):
        self.result_signal.emit(self.task_key, message)

    def _build_handler_params(self, operation_cls: type) -> dict[str, object]:
        return build_task_params(self.descriptor.task_spec, self.settings, operation_cls)

    def _handle_finished(self):
        success = self._run_error is None
        self.running_changed.emit(self.task_key, False)
        if success:
            message = '任务执行完成。'
        else:
            message = f'任务执行失败: {self._run_error}'
        self.completed.emit(self.task_key, success, message)

    def run(self):
        self._run_error = None
        handler_object = None
        try:
            operation_cls = TaskLoader.load_class(self.descriptor.module_path, self.descriptor.class_name)
            params = self._build_handler_params(operation_cls)

            FilesBasic.set_bootstrap_reporter(self._forward_result)
            try:
                handler_object = operation_cls(**params)
            finally:
                FilesBasic.clear_bootstrap_reporter()

            handler_object.result_signal.connect(self._forward_result, Qt.QueuedConnection)
            handler_object.set_work_folder(self.work_folder)
            handler_object.selected_dirs_handler(self.wanted_items)
        except Exception as exc:
            self._run_error = str(exc)
            self.result_signal.emit(
                self.task_key,
                TaskMessage.build(f'任务执行失败: {self._run_error}', level=MessageLevel.ERROR),
            )
        finally:
            if handler_object is not None:
                handler_object.close_log_session()


def build_task_setting_lines(settings: AppSettings, descriptor: TaskDescriptor) -> list[str]:
    entries = settings.get_setting_entries('Batch_Files', group_name=descriptor.key)
    return [
        f"{humanize_setting_label(entry['path'][-1])}: {format_setting_value(entry['value'])}"
        for entry in entries
    ]


def format_setting_value(value) -> str:
    if isinstance(value, bool):
        return 'True' if value else 'False'
    return str(value)


def release_active_binding(active_bindings: dict[str, BatchFilesBinding], task_key: str, binding: BatchFilesBinding):
    if active_bindings.get(task_key) is binding:
        active_bindings.pop(task_key, None)
    binding.deleteLater()


def launch_batch_operation(window: MainWindow, active_bindings: dict[str, BatchFilesBinding], descriptor: TaskDescriptor):
    existing_binding = active_bindings.get(descriptor.key)
    if existing_binding is not None and existing_binding.isRunning():
        window.FileWindow.append_operation_log(
            descriptor.key,
            TaskMessage.build('任务仍在运行，请稍候。', level=MessageLevel.WARNING),
        )
        return

    work_folder, wanted_items = window.FileWindow.get_selected_directories()
    if not work_folder:
        message = '请先选择工作目录。'
        window.FileWindow.set_selection_status(message, is_error=True)
        window.FileWindow.append_operation_log(
            descriptor.key,
            TaskMessage.build(f'未执行：{message}', level=MessageLevel.WARNING),
        )
        window.FileWindow.notify_blocking_issue(message)
        return

    if not wanted_items:
        message = '请至少勾选一个待处理目录。'
        window.FileWindow.set_selection_status(message, is_error=True)
        window.FileWindow.append_operation_log(
            descriptor.key,
            TaskMessage.build(f'未执行：{message}', level=MessageLevel.WARNING),
        )
        window.FileWindow.notify_blocking_issue(message)
        return

    if not TaskExecutionConfirmDialog.confirm(
        task_title=descriptor.title,
        selected_dirs=wanted_items,
        settings_lines=build_task_setting_lines(window.settings, descriptor),
        parent=window.FileWindow,
    ):
        window.FileWindow.append_operation_log(
            descriptor.key,
            TaskMessage.build('未执行：已取消执行确认。', level=MessageLevel.INFO),
        )
        return

    binding = BatchFilesBinding(descriptor, window.settings, work_folder, wanted_items)
    binding.result_signal.connect(window.FileWindow.append_operation_log, Qt.QueuedConnection)
    binding.running_changed.connect(window.FileWindow.set_task_running, Qt.QueuedConnection)
    binding.completed.connect(window.FileWindow.finish_task, Qt.QueuedConnection)
    binding.completed.connect(
        lambda *_args, task_key=descriptor.key, current_binding=binding: release_active_binding(
            active_bindings,
            task_key,
            current_binding,
        ),
        Qt.QueuedConnection,
    )

    active_bindings[descriptor.key] = binding
    window.FileWindow.set_selection_status(f'已准备执行，当前勾选了 {len(wanted_items)} 个目录。')
    window.FileWindow.log_operation_start(descriptor.key, work_folder, wanted_items)
    window.FileWindow.append_operation_log(
        descriptor.key,
        TaskMessage.build('正在加载任务实现...', level=MessageLevel.INFO),
    )
    binding.running_changed.emit(descriptor.key, True)
    binding.start()


def register_batch_operation(window: MainWindow, active_bindings: dict[str, BatchFilesBinding], descriptor: TaskDescriptor):
    window.FileWindow.register_task(
        descriptor,
        lambda current_descriptor=descriptor: launch_batch_operation(window, active_bindings, current_descriptor),
        has_settings=window.SettingWindow.has_task_settings(descriptor.key),
        open_settings_callback=lambda task_key=descriptor.key: window.open_task_settings(task_key),
    )


def order_task_descriptors(settings: AppSettings, descriptors: list[TaskDescriptor]) -> list[TaskDescriptor]:
    descriptor_map = {descriptor.key: descriptor for descriptor in descriptors}
    ordered_keys = settings.get_task_order([descriptor.key for descriptor in descriptors])
    return [
        descriptor_map[key]
        for key in ordered_keys
        if key in descriptor_map
    ]

def main():
    app = QApplication(sys.argv)
    settings = AppSettings()
    settings.load_settings()
    task_descriptors = order_task_descriptors(settings, build_task_descriptors())
    apply_app_theme(settings.theme, app)

    window = MainWindow(settings, task_descriptors)
    active_bindings: dict[str, BatchFilesBinding] = {}

    for descriptor in task_descriptors:
        register_batch_operation(window, active_bindings, descriptor)

    window.show_for_launch()
    for warning in settings.consume_startup_warnings():
        QTimer.singleShot(0, lambda message=warning: window.show_notification('warning', '任务顺序', message, duration=5000))
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
