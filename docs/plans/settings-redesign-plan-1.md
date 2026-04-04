# 任务注册与 CLI 收口重构设计记录

## Summary
- 保留现有 GUI 懒加载：启动期只加载轻量注册信息，执行时才动态导入任务实现。
- 把任务注册拆成两层：
  - 核心层 `TaskSpec`：无 GUI 依赖，供 GUI 和 CLI 共用。
  - 展示层 `TaskUiMeta`：只承载 `icon` 等 GUI 元数据。
- 用共享参数构建器统一 GUI/CLI 的配置拼装与校验，替换当前 `main.py` 里的分散逻辑。
- 用共享 CLI runner 替换各任务模块底部重复的交互式 `main()`。
- `Batch_Files` 配置直接切到 `task_key` 分组；不做旧分组兼容、不做迁移、不做告警。

## Implementation Changes
### 1. 任务注册分层
- 新增核心 registry 模块，定义 `TaskSpec`：
  - 字段固定为 `key`、`title`、`description`、`module_path`、`class_name`、`default_params`。
  - 不包含 `icon`、`FIF`、Qt 或任何 UI 依赖。
- 新增 GUI 展示元数据模块，定义 `TaskUiMeta`：
  - 至少包含 `task_key` 和 `icon`。
  - 由 GUI 层把 `TaskSpec + TaskUiMeta` 组装成现有 UI 可用的 descriptor。
- `main.py` 不再持有任务注册真相源，只消费 `core` registry 和 GUI meta。
- CLI 只消费核心 registry，不导入 GUI descriptor 或 `qfluentwidgets`。

### 2. 统一参数构建
- 新增共享函数：`build_task_params(task_spec, settings, operation_cls, overrides=None) -> dict[str, object]`。
- 参数来源和优先级固定为：`overrides` > `Batch_Files.<task_key>` > `task_spec.default_params` > `__init__` 默认值。
- 该函数负责：
  - 从 `AppSettings` 读取整组 `Batch_Files.<task_key>` 配置。
  - 合并默认参数。
  - 用 `inspect.signature(operation_cls.__init__)` 校验未知键和缺失必填参数。
  - 输出最终 `kwargs`，GUI 和 CLI 共用。
- `main.py` 的执行链改为调用这个共享函数，GUI 和 CLI 共用同一份参数拼装规则。
- `AppSettings` 收口为“配置读写与设置页支持”：
  - 增加按 `(category, group_name)` 读取整组配置的通用 API。
  - `Batch_Files_Settingmap` 全部改为 `task_key` 路径。
- 设置页、确认弹窗、任务设置入口统一按 `task_key` 取配置。

### 3. 统一 CLI runner
- 新增共享 CLI runner，例如 `run_task_cli(task_key: str, operation_cls: type | None = None) -> int`。
- runner 统一负责：
  - 加载 `AppSettings`
  - 读取 `TaskSpec`
  - 解析并构建参数
  - 询问工作目录
  - 列出一级子目录并接收选择
  - 实例化并执行任务
  - 处理错误码和终端输出
- 每个任务模块的 `if __name__ == '__main__':` 改成薄包装，只委托给共享 runner。
- 模块包装保留 `python -m modules.gen_subtitles` 这种入口，但不再维护自己的交互流程。
- 为避免 `python -m` 自执行场景的重复导入，模块包装把当前任务类显式传给 runner。
- 删除各任务模块底部重复的交互式 `main()` 代码。
- 删除任务模块里的 `sys.path.append(...)`，统一使用包导入。
- CLI 规范只支持 `python -m modules.<task_module>`；README 和开发文档同步更新，不再描述“直接执行脚本文件”。

### 4. 配置结构与任务实现边界
- `configs/settings.json` 中 `Batch_Files` 的任务分组统一改成稳定 `task_key`，例如 `subtitle-generation`、`dicom-processing`。
- 任务实现类不再读取应用层配置文件：
  - 删除 `GenSubtitles` 回读仓库内 `configs/settings.json` 的逻辑。
  - 保留纯运行时 fallback，例如默认模型路径、本地依赖检查，但这些 fallback 不再触达应用配置文件。
- GUI 和 CLI 都必须通过共享参数构建器把配置传入任务类；任务类自身只消费显式参数。
- `TaskSpec` 成为任务运行身份的唯一来源。

## Test Plan
- 核心 registry 测试：
  - `TaskSpec.key` 唯一。
  - 按 key 可查到 spec。
  - 导入核心 registry 时不导入 `ui` 或 `qfluentwidgets`。
- 参数构建器测试：
  - `Batch_Files.<task_key>` 正确覆盖 `default_params`。
  - `default_params` 能补齐缺失值。
  - 未知配置键报明确错误。
  - 缺失必填参数报明确错误。
- `AppSettings` 测试：
  - 能按 `task_key` 读取整组任务配置。
  - 设置页相关查询仍能正确列出该任务设置项。
- CLI runner 测试：
  - 正常交互路径返回 0。
  - 非法目录、非法序号、任务执行失败返回非 0。
  - runner 实际调用共享参数构建器，而不是模块自定义逻辑。
- 模块包装 smoke test：
  - 至少验证一个 `python -m modules.<task_module>` 风格入口委托到共享 runner。
- GUI 回归测试：
  - 任务顺序、设置入口、确认弹窗显示的参数仍正确。
  - GUI 执行时仍然通过 `TaskLoader` 懒加载任务类。

## Docs Updates
- 更新 `README.md`：
  - CLI 用法改为 `python -m modules.<task_module>`。
  - 明确 GUI 和 CLI 共用同一份用户配置。
- 更新 `docs/dev/lazy-load-design.md`：
  - registry 拆成核心层和 GUI 展示层。
  - CLI 复用核心 registry 和参数构建器，但不引入 GUI 依赖。
- 更新 `docs/dev/batch-processing-backend-guide.md`：
  - 新增共享参数构建器和共享 CLI runner 约定。
  - 删除“任务自己维护单独 `main()`”的旧说法。
- 更新 `docs/dev/fluent-ui-design.md`：
  - 改为“GUI descriptor 由 `TaskSpec + TaskUiMeta` 组装”。
- 更新 `docs/plans/settings-redesign-plan.md`，使其与最终实现方案一致，避免继续建议“直接把 `build_task_descriptors()` 平移”。

## Assumptions
- 这次不引入每任务专属 config dataclass；运行时仍使用共享构建器输出的 `dict kwargs`。
- `title` 和 `description` 保留在核心 `TaskSpec`，因为它们对 CLI 帮助文本也有价值；仅 `icon` 下沉到 GUI 展示层。
- CLI 可以继续依赖当前环境已安装的 PySide6，但实现上不应因为任务注册而额外导入 GUI 展示依赖。
- 不做旧配置兼容、迁移或过期告警；仓库默认配置和用户配置都直接切到新结构。
