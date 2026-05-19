# RD-tools

[English](README.md) | [简体中文](README_ZH.md)

RD-tools is a modular desktop utility for repetitive research-and-development workflows. It provides a PySide6 GUI, shared task settings, and optional CLI entry points for individual batch-processing tasks.

## Features

- **Unified task center**: select one working directory, choose subdirectories, and run registered batch tasks from one UI.
- **Persistent settings**: GUI and CLI share `~/Develop/RD-tools-configs/settings.json`.
- **Task-specific configuration**: tasks expose typed settings through a schema-driven settings page.
- **Bilingual UI**: English source strings with optional Simplified Chinese Qt translations.
- **Packaging support**: PyInstaller build flow with resource bundling for styles, configs, and translations.

## Available tasks

### DICOM Processing

Converts DICOM series into images and, when needed, videos. Select the parent directory that contains DICOM data folders, choose target subdirectories, run **DICOM Processing**, and monitor the task log.

### ECG Signal Processing

Processes single-lead ECG CSV data and generates raw, filtered, and advanced analysis charts. Set the sampling rate to match your data before running the task.

### Bilibili Video Export

Repairs and merges cached Bilibili app video files into playable MP4 files.

### Subtitle Generation

Extracts audio from media files and generates SRT subtitles with a local Whisper-compatible toolchain. Optional transcription dependencies are excluded from the default packaged build.

### RGB Channel Split / Merge

Splits RGB image channels or merges channel pairs such as R/G/B fluorescence images into composite color results.

### Image Perspective Transform

Applies a preset quadrilateral perspective transform to images.

### Batch Rename

Renames files in bulk with prefix, full-name, body, or between-boundary matching rules.

### Mac Cleaner

Removes common macOS metadata and junk files from selected directories.

## Install and run from source

1. Install [Python](https://www.python.org/downloads/), [ffmpeg](https://www.ffmpeg.org/download.html), and [git](https://git-scm.com/downloads).
2. Clone the repository:

```bash
git clone https://github.com/waaall/RD-tools.git
cd RD-tools
```

3. Install runtime dependencies:

```bash
python install.py install-runtime
```

4. Start the GUI:

```bash
python main.py
```

You can also run selected task modules from the terminal, for example:

```bash
python -m modules.gen_subtitles
```

## Build executable

Create or update the isolated build environment:

```bash
python install.py setup-build-env
```

Compile translations manually when needed:

```bash
python install.py compile-translations
```

Build the executable:

```bash
python install.py build
```

`python install.py build` compiles `i18n/*.ts` into `.qm` before invoking PyInstaller. The build bundles `ui/qss`, `configs`, and `i18n` resources.

Optional transcription dependencies are skipped by default. To include them in a packaged build, run:

```bash
RD_TOOLS_INCLUDE_TRANSCRIPTION=1 python install.py build
```

## Update

```bash
git pull
```

If you have local changes, resolve any Git conflicts before running or building the app.

## Documentation

Open the Help page in the app for task manuals and developer notes. The in-app manuals are intentionally kept as existing Markdown documents for now.

Developer design notes live under `docs/dev/`, including the Fluent UI guide and the i18n design guide.
