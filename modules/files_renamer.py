from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from core import MessageLevel
from modules.files_basic import FilesBasic


READY_STATUS = "ready"


class MatchMode(str, Enum):
    PREFIX = "prefix"
    ALL = "all"
    BODY = "body"
    BETWEEN = "between"


class Config:
    DEFAULT_PATTERN = "-"
    DEFAULT_START_PATTERN = ""
    DEFAULT_END_PATTERN = ""
    DEFAULT_MODE = MatchMode.PREFIX.value
    DEFAULT_REPLACE_WITH = ""
    DEFAULT_INCLUDE_EXTENSION = False
    DEFAULT_CASE_SENSITIVE = False
    DEFAULT_RECURSIVE = False
    DEFAULT_WORKERS = 4
    DEFAULT_LOG_FOLDER_NAME = "files_renamer_log"


class RenameStatus:
    SUCCESS = "success"
    SKIPPED_NO_MATCH = "skipped_no_match"
    SKIPPED_MULTI_MATCH = "skipped_multi_match"
    SKIPPED_CONFLICT = "skipped_conflict"
    SKIPPED_EMPTY_NAME = "skipped_empty_name"
    SKIPPED_NO_CHANGE = "skipped_no_change"
    ERROR = "error"


@dataclass(frozen=True)
class FileNameParts:
    name_without_extension: str
    extension: str

    def compose(self, new_name_without_extension: str) -> str:
        return f"{new_name_without_extension}{self.extension}"


class FileNameParser:
    @staticmethod
    def split(file_name: str) -> FileNameParts:
        dot_index = file_name.rfind(".")
        if dot_index <= 0:
            return FileNameParts(name_without_extension=file_name, extension="")
        return FileNameParts(
            name_without_extension=file_name[:dot_index],
            extension=file_name[dot_index:],
        )


@dataclass(frozen=True)
class RenameRule:
    mode: MatchMode
    replace_with: str
    include_extension: bool
    case_sensitive: bool
    pattern: str = ""
    start_pattern: str = ""
    end_pattern: str = ""

    def build_new_name(self, old_file_name: str) -> tuple[str | None, str]:
        parts = FileNameParser.split(old_file_name)
        source_text = old_file_name if self.include_extension else parts.name_without_extension
        transformed_text, status = self._transform_text(source_text)

        if status != READY_STATUS:
            return None, status

        new_file_name = transformed_text if self.include_extension else parts.compose(transformed_text)
        if new_file_name == "":
            return None, RenameStatus.SKIPPED_EMPTY_NAME
        if new_file_name == old_file_name:
            return None, RenameStatus.SKIPPED_NO_CHANGE
        return new_file_name, READY_STATUS

    def _transform_text(self, text: str) -> tuple[str, str]:
        if self.mode == MatchMode.PREFIX:
            return self._replace_prefix(text)
        if self.mode == MatchMode.ALL:
            return self._replace_all(text)
        if self.mode == MatchMode.BODY:
            return self._replace_body(text)
        if self.mode == MatchMode.BETWEEN:
            return self._replace_between(text)
        raise ValueError(f"不支持的匹配模式: {self.mode}")

    def _replace_prefix(self, text: str) -> tuple[str, str]:
        if not self._starts_with(text, self.pattern):
            return text, RenameStatus.SKIPPED_NO_MATCH
        return f"{self.replace_with}{text[len(self.pattern):]}", READY_STATUS

    def _replace_all(self, text: str) -> tuple[str, str]:
        replaced_text, replacement_count = self._replace_all_occurrences(text)
        if replacement_count == 0:
            return text, RenameStatus.SKIPPED_NO_MATCH
        return replaced_text, READY_STATUS

    def _replace_body(self, text: str) -> tuple[str, str]:
        if self._starts_with(text, self.pattern):
            preserved_prefix = text[: len(self.pattern)]
            transformed_suffix, replacement_count = self._replace_all_occurrences(text[len(self.pattern):])
            if replacement_count == 0:
                return text, RenameStatus.SKIPPED_NO_MATCH
            return f"{preserved_prefix}{transformed_suffix}", READY_STATUS
        return self._replace_all(text)

    def _replace_between(self, text: str) -> tuple[str, str]:
        start_positions = self._find_all_substring_positions(text, self.start_pattern, allow_overlap=True)
        end_positions = self._find_all_substring_positions(text, self.end_pattern, allow_overlap=True)

        if not start_positions or not end_positions:
            return text, RenameStatus.SKIPPED_NO_MATCH
        if len(start_positions) > 1 or len(end_positions) > 1:
            return text, RenameStatus.SKIPPED_MULTI_MATCH

        start_index = start_positions[0]
        end_index = end_positions[0]
        start_end_index = start_index + len(self.start_pattern)
        if end_index < start_end_index:
            return text, RenameStatus.SKIPPED_NO_MATCH

        return f"{text[:start_end_index]}{self.replace_with}{text[end_index:]}", READY_STATUS

    def _replace_all_occurrences(self, text: str) -> tuple[str, int]:
        start = 0
        pieces: list[str] = []
        replacement_count = 0

        while True:
            match_index = self._find_substring(text, self.pattern, start)
            if match_index == -1:
                pieces.append(text[start:])
                break

            pieces.append(text[start:match_index])
            pieces.append(self.replace_with)
            start = match_index + len(self.pattern)
            replacement_count += 1

        return "".join(pieces), replacement_count

    def _starts_with(self, text: str, pattern: str) -> bool:
        if self.case_sensitive:
            return text.startswith(pattern)
        return text.lower().startswith(pattern.lower())

    def _find_substring(self, text: str, pattern: str, start: int) -> int:
        if self.case_sensitive:
            return text.find(pattern, start)
        return text.lower().find(pattern.lower(), start)

    def _find_all_substring_positions(
        self,
        text: str,
        pattern: str,
        allow_overlap: bool = False,
    ) -> list[int]:
        if not pattern:
            return []

        search_text = text if self.case_sensitive else text.lower()
        search_pattern = pattern if self.case_sensitive else pattern.lower()
        positions: list[int] = []
        search_start = 0
        step = 1 if allow_overlap else len(pattern)

        while True:
            match_index = search_text.find(search_pattern, search_start)
            if match_index == -1:
                return positions

            positions.append(match_index)
            search_start = match_index + step


@dataclass(frozen=True)
class BatchRenameConfig:
    target_dir: str
    rule: RenameRule
    recursive: bool
    max_workers: int


@dataclass(frozen=True)
class RenameTask:
    old_path: str
    new_path: str
    old_display_name: str
    new_display_name: str

    @property
    def source_key(self) -> str:
        return normalize_path_key(self.old_path)

    @property
    def target_key(self) -> str:
        return normalize_path_key(self.new_path)


@dataclass(frozen=True)
class RenameSummary:
    root_dir: str
    file_count: int
    stats: dict[str, int]


def normalize_path_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


class BatchRenameEngine:
    def __init__(
        self,
        config: BatchRenameConfig,
        message_callback: Callable[[str, MessageLevel], None] | None = None,
    ):
        self.config = config
        self.target_dir = os.path.abspath(config.target_dir)
        self._message_callback = message_callback

    def run(self) -> RenameSummary:
        if not os.path.isdir(self.target_dir):
            raise ValueError(f"指定的文件夹不存在: {self.target_dir}")

        self._validate_rule()
        file_paths = self._collect_file_paths()
        stats = self._create_stats()
        rename_tasks = self._build_rename_tasks(file_paths, stats)
        self._execute_rename_tasks(rename_tasks, stats)
        return RenameSummary(root_dir=self.target_dir, file_count=len(file_paths), stats=stats)

    def _emit(self, text: str, level: MessageLevel):
        if self._message_callback is not None:
            self._message_callback(text, level)

    def _validate_rule(self):
        if self.config.rule.mode == MatchMode.BETWEEN:
            if not self.config.rule.start_pattern or not self.config.rule.end_pattern:
                raise ValueError("between 模式下必须同时提供 start_pattern 和 end_pattern。")
            return

        if not self.config.rule.pattern:
            raise ValueError(f"{self.config.rule.mode.value} 模式下 pattern 不能为空。")

    def _create_stats(self) -> dict[str, int]:
        return {
            RenameStatus.SUCCESS: 0,
            RenameStatus.SKIPPED_NO_MATCH: 0,
            RenameStatus.SKIPPED_MULTI_MATCH: 0,
            RenameStatus.SKIPPED_CONFLICT: 0,
            RenameStatus.SKIPPED_EMPTY_NAME: 0,
            RenameStatus.SKIPPED_NO_CHANGE: 0,
            RenameStatus.ERROR: 0,
        }

    def _collect_file_paths(self) -> list[str]:
        file_paths: list[str] = []

        if self.config.recursive:
            for root, dir_names, file_names in os.walk(self.target_dir):
                dir_names[:] = [dir_name for dir_name in dir_names if not dir_name.startswith(".")]
                for file_name in file_names:
                    if file_name.startswith("."):
                        continue
                    file_paths.append(os.path.join(root, file_name))
            return file_paths

        with os.scandir(self.target_dir) as entries:
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                if entry.is_file():
                    file_paths.append(entry.path)

        return file_paths

    def _build_rename_tasks(
        self,
        file_paths: list[str],
        stats: dict[str, int],
    ) -> list[RenameTask]:
        existing_paths = {normalize_path_key(path) for path in file_paths}
        candidate_tasks: list[RenameTask] = []

        for file_path in file_paths:
            old_file_name = os.path.basename(file_path)
            new_file_name, status = self.config.rule.build_new_name(old_file_name)
            if status != READY_STATUS:
                stats[status] += 1
                self._log_skip_status(file_path, status)
                continue

            new_path = os.path.join(os.path.dirname(file_path), new_file_name)
            candidate_tasks.append(
                RenameTask(
                    old_path=file_path,
                    new_path=new_path,
                    old_display_name=self._display_path(file_path),
                    new_display_name=self._display_path(new_path),
                )
            )

        target_counter = Counter(task.target_key for task in candidate_tasks)
        approved_tasks: list[RenameTask] = []

        for task in candidate_tasks:
            if target_counter[task.target_key] > 1:
                stats[RenameStatus.SKIPPED_CONFLICT] += 1
                self._emit(
                    f"跳过重名冲突: '{task.old_display_name}' -> '{task.new_display_name}'，多个文件会生成相同名称。",
                    MessageLevel.WARNING,
                )
                continue

            if task.target_key in existing_paths and task.target_key != task.source_key:
                stats[RenameStatus.SKIPPED_CONFLICT] += 1
                self._emit(
                    f"跳过重名冲突: '{task.old_display_name}' -> '{task.new_display_name}'，目标文件已存在。",
                    MessageLevel.WARNING,
                )
                continue

            approved_tasks.append(task)

        return approved_tasks

    def _log_skip_status(self, file_path: str, status: str):
        display_name = self._display_path(file_path)
        if status == RenameStatus.SKIPPED_MULTI_MATCH:
            self._emit(
                f"跳过多重边界匹配: '{display_name}' 中开始或结束边界出现多次。",
                MessageLevel.WARNING,
            )
            return
        if status == RenameStatus.SKIPPED_EMPTY_NAME:
            self._emit(
                f"跳过空文件名结果: '{display_name}' 替换后文件名为空。",
                MessageLevel.WARNING,
            )

    def _execute_rename_tasks(
        self,
        rename_tasks: list[RenameTask],
        stats: dict[str, int],
    ):
        if not rename_tasks:
            return

        max_workers = max(1, min(self.config.max_workers, len(rename_tasks)))
        if max_workers == 1:
            for task in rename_tasks:
                stats[self._rename_single_file(task)] += 1
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for status in executor.map(self._rename_single_file, rename_tasks):
                stats[status] += 1

    def _rename_single_file(self, task: RenameTask) -> str:
        try:
            os.rename(task.old_path, task.new_path)
            return RenameStatus.SUCCESS
        except Exception as exc:
            self._emit(
                f"重命名失败: '{task.old_display_name}' -> '{task.new_display_name}'，原因: {exc}",
                MessageLevel.ERROR,
            )
            return RenameStatus.ERROR

    def _display_path(self, path: str) -> str:
        return os.path.relpath(path, self.target_dir)


class FilesRenamer(FilesBasic):
    def __init__(
        self,
        log_folder_name: str = Config.DEFAULT_LOG_FOLDER_NAME,
        mode: str = Config.DEFAULT_MODE,
        pattern: str = Config.DEFAULT_PATTERN,
        start_pattern: str = Config.DEFAULT_START_PATTERN,
        end_pattern: str = Config.DEFAULT_END_PATTERN,
        replace_with: str = Config.DEFAULT_REPLACE_WITH,
        include_extension: bool = Config.DEFAULT_INCLUDE_EXTENSION,
        case_sensitive: bool = Config.DEFAULT_CASE_SENSITIVE,
        recursive: bool = Config.DEFAULT_RECURSIVE,
        max_threads: int = Config.DEFAULT_WORKERS,
    ):
        super().__init__(
            log_folder_name=log_folder_name,
            out_dir_prefix="",
            max_threads=Config.DEFAULT_WORKERS,
            parallel=False,
        )
        self.mode = mode
        self.pattern = pattern
        self.start_pattern = start_pattern
        self.end_pattern = end_pattern
        self.replace_with = replace_with
        self.include_extension = include_extension
        self.case_sensitive = case_sensitive
        self.recursive = recursive
        self.max_threads = max_threads
        self._current_rule: RenameRule | None = None
        self._current_max_workers = Config.DEFAULT_WORKERS

    def selected_dirs_handler(self, indexs_list):
        try:
            self._current_rule = self._build_rule()
            self._current_max_workers = self._resolve_max_workers()
        except ValueError as exc:
            self.send_message(str(exc), level=MessageLevel.ERROR)
            return False
        return super().selected_dirs_handler(indexs_list)

    def _data_dir_handler(self, _data_dir: str):
        rule = self._current_rule or self._build_rule()
        config = BatchRenameConfig(
            target_dir=self._resolve_work_path(_data_dir),
            rule=rule,
            recursive=bool(self.recursive),
            max_workers=self._current_max_workers,
        )
        summary = BatchRenameEngine(config, message_callback=self.send_message).run()
        self.send_message(self._build_summary_text(_data_dir, summary), level=MessageLevel.INFO)

    def _build_rule(self) -> RenameRule:
        try:
            mode = MatchMode(str(self.mode))
        except ValueError as exc:
            raise ValueError(f"不支持的重命名模式: {self.mode}") from exc

        rule = RenameRule(
            mode=mode,
            replace_with=str(self.replace_with or ""),
            include_extension=bool(self.include_extension),
            case_sensitive=bool(self.case_sensitive),
            pattern=str(self.pattern or ""),
            start_pattern=str(self.start_pattern or ""),
            end_pattern=str(self.end_pattern or ""),
        )

        if mode == MatchMode.BETWEEN:
            if not rule.start_pattern or not rule.end_pattern:
                raise ValueError("between 模式下必须同时提供 start_pattern 和 end_pattern。")
        elif not rule.pattern:
            raise ValueError(f"{mode.value} 模式下 pattern 不能为空。")

        return rule

    def _resolve_max_workers(self) -> int:
        try:
            converted = int(self.max_threads)
        except (TypeError, ValueError):
            self.send_message(
                f"max_threads 配置无效，已使用默认值 {Config.DEFAULT_WORKERS}",
                level=MessageLevel.WARNING,
            )
            return Config.DEFAULT_WORKERS

        if converted <= 0:
            self.send_message(
                f"max_threads 必须大于 0，已使用默认值 {Config.DEFAULT_WORKERS}",
                level=MessageLevel.WARNING,
            )
            return Config.DEFAULT_WORKERS

        return converted

    def _build_summary_text(self, data_dir: str, summary: RenameSummary) -> str:
        stats = summary.stats
        return (
            f"目录重命名完成: {data_dir} | "
            f"扫描文件 {summary.file_count} 个 | "
            f"成功 {stats[RenameStatus.SUCCESS]} | "
            f"未匹配 {stats[RenameStatus.SKIPPED_NO_MATCH]} | "
            f"多重匹配 {stats[RenameStatus.SKIPPED_MULTI_MATCH]} | "
            f"重名冲突 {stats[RenameStatus.SKIPPED_CONFLICT]} | "
            f"结果为空 {stats[RenameStatus.SKIPPED_EMPTY_NAME]} | "
            f"名称未变化 {stats[RenameStatus.SKIPPED_NO_CHANGE]} | "
            f"失败 {stats[RenameStatus.ERROR]}"
        )
