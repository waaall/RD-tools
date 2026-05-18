# RD-tools

[English](README.md) | [简体中文](README_ZH.md)

RD-tools 是一个面向研发流程中重复性工作的模块化桌面工具。它提供 PySide6 图形界面、统一任务设置，以及部分任务的 CLI 入口。

## 功能概览

- **统一任务中心**：选择一个工作目录，勾选子目录，然后从同一界面运行已注册的批处理任务。
- **持久化设置**：GUI 和 CLI 共用 `~/Develop/RD-tools-configs/settings.json`。
- **任务独立配置**：任务通过 schema 暴露类型化设置，并在设置页中统一渲染。
- **双语界面**：代码源字符串为英文，简体中文通过 Qt 翻译文件提供。
- **打包支持**：PyInstaller 构建流程会分发样式、默认配置和翻译资源。

## 可用任务

### DICOM Processing

将 DICOM 序列导出为图片，并在需要时生成视频。选择包含 DICOM 数据文件夹的上级目录，勾选目标子目录后运行 **DICOM Processing**，并在任务日志中查看进度。

### ECG Signal Processing

处理单导联 ECG CSV 数据，生成原始、滤波和高级分析图表。运行前请将采样率设置为与数据一致。

### Bilibili Video Export

修复并合并 Bilibili App 缓存视频，导出为可播放的 MP4 文件。

### Subtitle Generation

从媒体文件抽取音频，并使用本地 Whisper 兼容工具链生成 SRT 字幕。默认打包不包含体积较大的转写依赖。

### RGB Channel Split / Merge

拆分 RGB 图像通道，或将 R/G/B 等通道图像合成为彩色结果图。

### Image Perspective Transform

按预设四边形参数对图片执行透视变换。

### Batch Rename

按 prefix、all、body、between 等匹配规则批量重命名文件。

### Mac Cleaner

清理所选目录中的常见 macOS 元数据和垃圾文件。

## 从源码安装和运行

1. 安装 [Python](https://www.python.org/downloads/)、[ffmpeg](https://www.ffmpeg.org/download.html) 和 [git](https://git-scm.com/downloads)。
2. 克隆仓库：

```bash
git clone https://github.com/waaall/RD-tools.git
cd RD-tools
```

3. 安装运行时依赖：

```bash
python install.py install-runtime
```

4. 启动 GUI：

```bash
python main.py
```

也可以从终端运行部分任务模块，例如：

```bash
python -m modules.gen_subtitles
```

## 构建可执行程序

创建或更新隔离的构建环境：

```bash
python install.py setup-build-env
```

需要时可单独编译翻译文件：

```bash
python install.py compile-translations
```

构建可执行程序：

```bash
python install.py build
```

`python install.py build` 会在调用 PyInstaller 前把 `i18n/*.ts` 编译为 `.qm`。构建产物会携带 `ui/qss`、`configs` 和 `i18n` 资源。

默认打包会跳过可选的字幕转写依赖。如需包含它们，请运行：

```bash
RD_TOOLS_INCLUDE_TRANSCRIPTION=1 python install.py build
```

## 更新

```bash
git pull
```

如果你有本地改动，请先处理 Git 冲突，再运行或构建应用。

## 文档

任务手册和开发说明请在应用内 Help 页面查看。应用内手册本次仍沿用现有 Markdown 文档。
