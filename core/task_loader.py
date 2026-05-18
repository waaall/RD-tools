from __future__ import annotations

import importlib
import threading

from core.task_errors import TaskErrorCode, TaskUserError


class TaskLoader:
    _class_cache: dict[tuple[str, str], type] = {}
    _cache_lock = threading.Lock()

    @classmethod
    def load_class(cls, module_path: str, class_name: str) -> type:
        cache_key = (module_path, class_name)
        with cls._cache_lock:
            cached_class = cls._class_cache.get(cache_key)
            if cached_class is not None:
                return cached_class

            try:
                module = importlib.import_module(module_path)
            except Exception as exc:
                raise TaskUserError(
                    TaskErrorCode.MODULE_LOAD_FAILED,
                    {'module_path': module_path},
                    detail=str(exc),
                ) from exc
            try:
                task_class = getattr(module, class_name)
            except AttributeError as exc:
                raise TaskUserError(
                    TaskErrorCode.CLASS_NOT_FOUND,
                    {'module_path': module_path, 'class_name': class_name},
                    detail=str(exc),
                ) from exc

            cls._class_cache[cache_key] = task_class
            return task_class
