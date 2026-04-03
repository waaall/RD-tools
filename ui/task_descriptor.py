from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TaskDescriptor:
    key: str
    title: str
    description: str
    icon: Any
    module_path: str
    class_name: str
    settings_group: str
    default_params: dict[str, Any] = field(default_factory=dict)
