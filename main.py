from __future__ import annotations

import sys

from PySide6.QtCore import QCoreApplication, QThread, Qt, QTimer, Signal
from PySide6.QtWidgets import QApplication

from core import MessageLevel, TaskLoader, TaskMessage, format_task_exception
from core.i18n import TranslatorBundle
from core.setting_texts import translate_setting_text
from core.task_params import build_task_params
from main_window import MainWindow
from modules.app_settings import AppSettings
from modules.files_basic import FilesBasic
from ui import TaskDescriptor, apply_app_theme
from ui.task_ui_registry import build_task_descriptors
from widgets.confirm_dialog import TaskExecutionConfirmDialog


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
            message = self.tr('Task completed.')
        else:
            message = self.tr('Task failed: {0}').format(self._run_error)
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
            self._run_error = format_task_exception(exc)
            self.result_signal.emit(
                self.task_key,
                TaskMessage.build(self.tr('Task failed: {0}').format(self._run_error), level=MessageLevel.ERROR),
            )
        finally:
            if handler_object is not None:
                handler_object.close_log_session()


def build_task_setting_lines(settings: AppSettings, descriptor: TaskDescriptor) -> list[str]:
    entries = settings.get_setting_entries('Batch_Files', group_name=descriptor.key)
    return [
        f"{translate_setting_text(entry['path'][-1])}: {format_setting_value(entry['value'])}"
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


def refresh_language_guard(window: MainWindow, active_bindings: dict[str, BatchFilesBinding], pending_running: bool = False):
    """根据任务运行状态启停语言下拉，避免任务运行期间修改全局配置。"""
    has_running_task = pending_running or any(binding.isRunning() for binding in active_bindings.values())
    window.SettingWindow.set_language_change_enabled(not has_running_task)


def launch_batch_operation(window: MainWindow, active_bindings: dict[str, BatchFilesBinding], descriptor: TaskDescriptor):
    existing_binding = active_bindings.get(descriptor.key)
    if existing_binding is not None and existing_binding.isRunning():
        window.FileWindow.append_operation_log(
            descriptor.key,
            TaskMessage.build(
                QCoreApplication.translate('TaskRunner', 'A task is still running, please wait.'),
                level=MessageLevel.WARNING,
            ),
        )
        return

    work_folder, wanted_items = window.FileWindow.get_selected_directories()
    if not work_folder:
        message = QCoreApplication.translate('TaskRunner', 'Select a working directory first.')
        window.FileWindow.set_selection_status(message, is_error=True)
        window.FileWindow.append_operation_log(
            descriptor.key,
            TaskMessage.build(
                QCoreApplication.translate('TaskRunner', 'Not run: {0}').format(message),
                level=MessageLevel.WARNING,
            ),
        )
        window.FileWindow.notify_blocking_issue(message)
        return

    if not wanted_items:
        message = QCoreApplication.translate('TaskRunner', 'Select at least one directory to process.')
        window.FileWindow.set_selection_status(message, is_error=True)
        window.FileWindow.append_operation_log(
            descriptor.key,
            TaskMessage.build(
                QCoreApplication.translate('TaskRunner', 'Not run: {0}').format(message),
                level=MessageLevel.WARNING,
            ),
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
            TaskMessage.build(
                QCoreApplication.translate('TaskRunner', 'Not run: run confirmation was cancelled.'),
                level=MessageLevel.INFO,
            ),
        )
        return

    binding = BatchFilesBinding(descriptor, window.settings, work_folder, wanted_items)
    binding.result_signal.connect(window.FileWindow.append_operation_log, Qt.QueuedConnection)
    binding.running_changed.connect(window.FileWindow.set_task_running, Qt.QueuedConnection)
    binding.running_changed.connect(
        lambda _task_key, is_running, current_window=window: refresh_language_guard(
            current_window,
            active_bindings,
            pending_running=is_running,
        ),
        Qt.QueuedConnection,
    )
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
    window.FileWindow.set_selection_status(
        QCoreApplication.translate('TaskRunner', 'Ready to run; {0} directories selected.').format(len(wanted_items))
    )
    window.FileWindow.log_operation_start(descriptor.key, work_folder, wanted_items)
    window.FileWindow.append_operation_log(
        descriptor.key,
        TaskMessage.build(
            QCoreApplication.translate('TaskRunner', 'Loading task implementation...'),
            level=MessageLevel.INFO,
        ),
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


class AppController:
    """应用级协调器:持有 app / settings / translator / window / active_bindings。"""

    def __init__(self, app: QApplication):
        self._app = app
        self.settings = AppSettings()
        self._translators = TranslatorBundle()
        self.window: MainWindow | None = None
        self._active_bindings: dict[str, BatchFilesBinding] = {}

    def start(self):
        """首次启动:装载 translator、构建并显示主窗口。"""
        self._translators.install(self._app, self.settings.language)
        self._build_window()
        self.window.show_for_launch()
        self._flush_startup_warnings()

    def _build_window(self):
        """构建 MainWindow,完成任务注册与语言切换信号连接。"""
        task_descriptors = order_task_descriptors(self.settings, build_task_descriptors())
        apply_app_theme(self.settings.theme, self._app)
        self.window = MainWindow(self.settings, task_descriptors)
        for descriptor in task_descriptors:
            register_batch_operation(self.window, self._active_bindings, descriptor)
        self.window.SettingWindow.language_changed.connect(self._change_language)

    def _change_language(self, language_mode: str):
        """语言设置写入配置文件,下次启动时生效。"""
        if language_mode == self.settings.language:
            return

        title = QCoreApplication.translate('AppController', 'Language')
        if self._has_running_task():
            content = QCoreApplication.translate(
                'AppController', 'Cannot switch language while a task is running.')
            self.window.show_notification('warning', title, content)
            self.window.SettingWindow.set_language_value(self.settings.language)
            return

        if not self.settings.save_settings('language', language_mode):
            content = QCoreApplication.translate(
                'AppController', 'Failed to save the language setting.')
            self.window.show_notification('error', title, content)
            self.window.SettingWindow.set_language_value(self.settings.language)
            return

        content = QCoreApplication.translate(
            'AppController',
            'Language setting saved. Restart the app to apply it.',
        )
        self.window.show_notification('success', title, content)

    def _flush_startup_warnings(self):
        """启动期配置告警延后到事件循环里弹出,避免与窗口构建抢时序。"""
        title = QCoreApplication.translate('AppController', 'Task order')
        for warning in self.settings.consume_startup_warnings():
            QTimer.singleShot(
                0,
                lambda message=warning: self.window.show_notification(
                    'warning', title, message, duration=5000),
            )

    def _has_running_task(self) -> bool:
        return any(binding.isRunning() for binding in self._active_bindings.values())


def main():
    app = QApplication(sys.argv)
    controller = AppController(app)
    controller.start()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
