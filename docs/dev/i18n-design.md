# RD-tools 多语言设计文档

本文档描述 RD-tools 的多语言架构、代码约定、翻译文件维护方式和打包注意事项。UI 层的视觉与页面组织仍以 `docs/dev/fluent-ui-design.md` 为准；本文只关注“用户可见文案如何进入翻译系统”。

## 1. 设计目标

当前多语言系统的目标是：

1. 英文作为源语言
   - 代码中的用户可见源字符串统一写英文。
   - 英文模式不安装应用 translator，直接显示源字符串。

2. 简体中文通过 Qt 翻译文件覆盖
   - 翻译源文件为 `i18n/rdtools_zh_CN.ts`。
   - 运行时加载编译产物 `i18n/rdtools_zh_CN.qm`。

3. 持久化值和显示文案分离
   - `settings.json` 中保存稳定 key，例如 `system`、`zh_CN`、`en`、`prefix`。
   - UI 显示时再把稳定 key 翻译成当前语言文案。

4. 语言切换下次启动生效
   - 修改语言设置后立即写入配置文件。
   - 当前窗口不重建，避免丢失任务中心的工作目录、勾选项和日志。

## 2. 核心模块

### 2.1 `core/i18n.py`

`core/i18n.py` 负责应用级语言逻辑：

- 定义稳定语言值：
  - `system`
  - `zh_CN`
  - `en`
- 兼容旧配置值，例如 `English`、`zh`、`简体中文`
- 根据语言模式解析实际 locale
- 持有并安装 `QTranslator`

英文是源语言，因此 `resolve_locale()` 得到 `en` 时不安装应用 translator。

### 2.2 `core/setting_texts.py`

设置系统中的 category、group、json_key、option 都是稳定 key，不应直接显示给用户。`core/setting_texts.py` 集中处理这些 key 的显示文案：

- `translate_setting_text(value)`
- `translate_option_label(setting_key, value)`

例如：

```python
translate_option_label('mode', 'prefix')
```

在中文环境下显示为“前缀”，但配置文件里仍保存 `prefix`。

### 2.3 `core/task_registry.py`

任务标题和任务说明是核心元数据，放在 `TaskSpec` 中。为了让 Qt 的 lupdate 能提取这些字符串，应使用 `QT_TRANSLATE_NOOP` 标记：

```python
TaskSpec(
    key='files-renamer',
    title=QT_TRANSLATE_NOOP('Tasks', 'Batch Rename'),
    description=QT_TRANSLATE_NOOP('Tasks', 'Batch rename files using prefix / all / body / between rules.'),
    module_path='modules.files_renamer',
    class_name='FilesRenamer',
)
```

`TaskDescriptor.title` 和 `TaskDescriptor.description` 再通过 `QCoreApplication.translate('Tasks', ...)` 得到当前语言文案。

`core/task_registry.py` 允许在缺少 PySide6 时导入：如果 `QT_TRANSLATE_NOOP` 不可用，会退化成返回原字符串。这样 `install.py install-runtime` 可以在运行时依赖尚未安装时正常启动。

### 2.4 `core/task_errors.py`

任务框架层错误使用稳定 code 表示，再在展示时翻译：

- `TaskErrorCode`
- `TaskUserError`
- `format_task_exception()`

底层异常 detail 原样保留，不尝试翻译。这样可以既给用户本地化的错误前缀，又保留真实调试信息。

## 3. 字符串使用约定

### 3.1 QObject 页面内文案

页面类、控件类和 QObject 子类里的静态文案优先使用：

```python
self.tr('English source text')
```

适用位置：

- 页面标题
- 按钮文案
- 空状态
- 本页面内部的提示文本

### 3.2 非 QObject 或跨页面文案

不方便使用 `self.tr()` 的位置使用：

```python
QCoreApplication.translate('ContextName', 'English source text')
```

适用位置：

- `main.py` 中的任务调度提示
- 任务框架错误文案
- 模块级 helper 返回的 UI 文案

context 要稳定、可读，避免随意变化。已有 context 包括：

- `AppController`
- `TaskRunner`
- `Tasks`
- `TaskErrors`
- `SettingsMeta`

### 3.3 稳定 key 不直接显示

以下值属于稳定 key，不应直接作为 UI 文案展示：

- task key，例如 `files-renamer`
- setting field id，例如 `rename_mode`
- settings json key，例如 `mode`
- option value，例如 `prefix`
- language value，例如 `zh_CN`

显示时应通过对应 helper 翻译。

### 3.4 业务处理模块消息

目前 `modules/*` 中仍有大量中文或英文业务消息。多语言支持本次优先覆盖应用框架、任务中心、设置页、帮助页和任务框架错误。

后续如果要继续国际化业务模块，应优先处理：

1. 任务运行前的框架提示
2. 用户可恢复的错误提示
3. 高频成功/警告消息
4. 详细 debug 信息

不建议一次性机械替换所有业务日志，因为部分日志同时承担调试信息角色。

## 4. 设置与语言模式

语言设置位于：

```json
{
    "General": {
        "language": "system"
    }
}
```

允许值来自 `LANGUAGE_OPTIONS`：

- `system`
- `zh_CN`
- `en`

旧值迁移由 `coerce_language()` 处理。无法识别的语言值回退到 `system`。

设置页中的语言卡片不走普通 `update_setting()`，而是发出 `language_changed` 信号，由 `AppController` 决定是否允许写入。任务运行期间会禁用语言切换，避免运行中修改全局配置。

语言设置保存成功后提示用户重启应用生效。

## 5. 翻译文件维护

### 5.1 文件角色

- `i18n/rdtools_zh_CN.ts`
  - 可读、可编辑的翻译源文件
- `i18n/rdtools_zh_CN.qm`
  - Qt 运行时加载的二进制翻译文件

两者都提交到仓库。`.qm` 需要随 `.ts` 更新，否则源码运行或打包产物可能加载到旧翻译。

### 5.2 编译翻译

手动编译：

```bash
python install.py compile-translations
```

打包时会自动编译：

```bash
python install.py build
```

构建脚本优先使用目标 Python 环境中的 `pyside6-lrelease`，避免误用系统环境里的 Qt 工具。

### 5.3 新增文案流程

新增或修改用户可见文案后：

1. 在代码中使用 `tr()`、`QCoreApplication.translate()` 或 `QT_TRANSLATE_NOOP`
2. 更新 `i18n/rdtools_zh_CN.ts`
3. 补齐对应中文翻译
4. 运行 `python install.py compile-translations`
5. 启动应用检查中英文显示
6. 运行相关测试

当前仓库没有专门封装 lupdate 命令。如果批量新增大量文案，建议先用 Qt Linguist / `pyside6-lupdate` 更新 `.ts`，再人工检查 context 和翻译内容。

## 6. 打包资源

打包时必须带上 `i18n` 目录。当前 `install.py` 的 `RESOURCE_DATA_DIRECTORIES` 包含：

- `ui/qss`
- `configs`
- `i18n`

`core/resource_paths.py` 会在源码运行和 PyInstaller 产物中查找资源路径。`TranslatorBundle` 通过：

```python
resolve_resource_path('i18n', f'rdtools_{locale}.qm')
```

定位应用翻译文件。

如果打包产物中文不生效，优先检查：

1. `i18n/rdtools_zh_CN.qm` 是否存在
2. PyInstaller 命令或 spec 是否包含 `i18n`
3. `resolve_resource_path()` 是否能在产物中找到该文件
4. 当前语言设置是否解析为 `zh_CN`

## 7. 测试建议

多语言相关改动至少验证：

1. `python install.py compile-translations`
2. `python -m unittest discover -s tests`
3. 模拟缺少 PySide6 时 `import install` 仍能成功
4. 中文模式下任务标题、设置页、任务确认弹窗能显示中文
5. 英文模式下不安装应用 translator，显示英文源字符串
6. 修改语言设置后不重建窗口，只提示重启生效
7. 任务运行期间语言下拉不可用
8. 打包产物包含 `i18n/*.qm`

## 8. 常见问题

### 8.1 为什么语言切换不是立即生效

Qt translator 可以运行时安装和卸载，但现有窗口中的大部分控件不会自动重新取一次文案。要立即生效通常需要重建窗口或手写大量刷新逻辑。

任务中心保存了用户当前工作目录、勾选目录、任务日志和运行态。为了不因为切换语言丢失这些临时状态，当前策略是“保存设置，下次启动生效”。

### 8.2 为什么英文不需要 `.qm`

英文是代码源字符串。缺少 translator 时 Qt 会显示源字符串，因此英文模式不需要额外翻译文件。

### 8.3 为什么不能直接翻译配置 key

配置 key 是持久化协议的一部分。翻译 key 会破坏旧配置兼容、任务参数构建和测试快照。

正确做法是：

- 存储值保持稳定
- UI 显示时翻译
- 保存时仍写回稳定值

### 8.4 为什么 install 脚本要避免顶层运行时依赖

`install.py install-runtime` 的职责是安装运行时依赖。如果它在导入阶段就依赖 PySide6、qfluentwidgets 或其他尚未安装的库，用户刚 clone 仓库时会无法运行安装命令。

因此 `install.py` 顶层应尽量只依赖标准库。需要项目模块时，优先在具体函数内部延迟导入。


### 8.4 QT_TRANSLATE_NOOP 和 tr 的区别？

`self.tr("English source text")` 和 `QT_TRANSLATE_NOOP("context", "English source text")` 都和 Qt 国际化（i18n）有关，但它们用途、翻译时机、上下文机制都不同。

可以理解为：

* `tr()`：**运行时翻译（立即翻译）**，适合 UI 直接显示
* `QT_TRANSLATE_NOOP()`：**仅标记待翻译（不立即翻译）**，适合静态/数据驱动场景，之后再用 `QCoreApplication.translate()` 延迟翻译。
