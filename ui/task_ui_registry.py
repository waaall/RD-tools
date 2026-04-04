from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qfluentwidgets import FluentIcon as FIF

from core.task_registry import get_task_specs
from ui.task_descriptor import TaskDescriptor


@dataclass(frozen=True, slots=True)
class TaskUiMeta:
    task_key: str
    icon: Any


_TASK_UI_META = {
    'merge-colors': TaskUiMeta(task_key='merge-colors', icon=FIF.APPLICATION),
    'dicom-processing': TaskUiMeta(task_key='dicom-processing', icon=FIF.FOLDER_ADD),
    'split-colors': TaskUiMeta(task_key='split-colors', icon=FIF.SYNC),
    'twist-images': TaskUiMeta(task_key='twist-images', icon=FIF.EDIT),
    'bilibili-export': TaskUiMeta(task_key='bilibili-export', icon=FIF.PLAY),
    'ecg-handler': TaskUiMeta(task_key='ecg-handler', icon=FIF.IOT),
    'subtitle-generation': TaskUiMeta(task_key='subtitle-generation', icon=FIF.DOCUMENT),
    'files-renamer': TaskUiMeta(task_key='files-renamer', icon=FIF.EDIT),
    'mac-cleaner': TaskUiMeta(task_key='mac-cleaner', icon=FIF.FOLDER),
}


def build_task_descriptors() -> list[TaskDescriptor]:
    descriptors: list[TaskDescriptor] = []
    for spec in get_task_specs():
        ui_meta = _TASK_UI_META[spec.key]
        descriptors.append(TaskDescriptor.from_spec(spec, ui_meta.icon))
    return descriptors
