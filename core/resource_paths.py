from __future__ import annotations

import sys
from pathlib import Path


def _candidate_resource_roots() -> list[Path]:
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass))

        executable_dir = Path(sys.executable).resolve().parent
        candidates.append(executable_dir)
        candidates.append(executable_dir / "_internal")

        if executable_dir.name == "MacOS":
            contents_dir = executable_dir.parent
            candidates.append(contents_dir / "Resources")
            candidates.append(contents_dir / "Frameworks")

    candidates.append(Path(__file__).resolve().parents[1])

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates


def resolve_resource_path(*parts: str) -> Path:
    attempted_paths: list[Path] = []
    for root in _candidate_resource_roots():
        candidate = root.joinpath(*parts)
        attempted_paths.append(candidate)
        if candidate.exists():
            return candidate

    attempted = ", ".join(str(path) for path in attempted_paths)
    wanted = "/".join(parts)
    raise FileNotFoundError(f"Resource not found: {wanted}. Tried: {attempted}")
