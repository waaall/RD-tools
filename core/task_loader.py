from __future__ import annotations

import importlib
import threading


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

            module = importlib.import_module(module_path)
            try:
                task_class = getattr(module, class_name)
            except AttributeError as exc:
                raise ImportError(f'模块 {module_path} 中不存在类 {class_name}') from exc

            cls._class_cache[cache_key] = task_class
            return task_class
