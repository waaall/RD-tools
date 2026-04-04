# Schema 驱动的设置系统重构计划

## Summary
- 把设置系统改成“schema 为真相源”，不再让 `AppSettings` 手写 `*_Settingmap` 和依赖 `configs/settings.json` 作为运行时定义。
- 任务设置 schema 与 `TaskSpec` 放在同一处维护；`Task_Center`、`General`、`Network` 也补成同一套 dataclass schema。
- `configs/settings.json` 改成由 schema 生成的仓库快照，只用于开发期对账；运行时默认值一律从 schema 生成。
- 配置文件解析/校验失败时，在设置页显示持久 warning 卡片，并提供“恢复默认配置”按钮；按钮只覆盖用户目录 `~/Develop/RD-tools-configs/settings.json`，点击前必须二次确认。

## Key Changes
### 1. 统一 schema 模型
- 新增设置 schema 层，定义最少两类 dataclass：
  - `SettingFieldSpec`：字段 id、category、group_key、json_key、default、value_type、options、label、description、visible。
  - `TaskSettingSpec`：任务专用字段定义，可被转换成 `SettingFieldSpec`。
- 扩展 `core/task_registry.py` 中的 `TaskSpec`：
  - 保留 `default_params`，只用于“非用户可配置”的运行时默认值。
  - 新增 `settings: tuple[TaskSettingSpec, ...]`，承载所有任务设置的默认值、类型、选项和 UI 展示信息。
- `Task_Center` 不再在 `AppSettings` 里硬编码；`task_order` 由 schema 定义，默认顺序从当前 task registry 动态生成。
- `General` / `Network` 也迁到同一套 schema，避免默认配置文件损坏时失去恢复来源。

### 2. `AppSettings` 瘦身为 facade + store
- `AppSettings` 不再维护 `General_Settingmap` / `Network_Settingmap` / `Batch_Files_Settingmap` / `Task_Center_Settingmap`。
- `AppSettings` 改为只做：
  - 读取用户配置文件
  - 读取仓库快照文件并做健康检查
  - 用 schema 生成默认配置
  - 合并用户 overrides，产出规范化后的 effective settings
  - 提供 `get_setting_entries()`、`get_group_values()`、`save_settings()`、`save_task_order()`、`reset_user_settings_to_defaults()` 等 facade API
  - 暴露 `ConfigHealth`
- 运行时加载顺序固定为：
  - 先从 schema 生成完整默认配置
  - 再尝试读取用户配置并覆盖
  - 再尝试读取仓库快照做校验，不作为默认值真相源
- 规范化/校验规则固定为：
  - JSON 解析失败：整份文件忽略，记录 health issue
  - 未知 category/group/key：忽略并记录 issue
  - 字符串到 bool/int/float 维持现有宽松转换
  - 类型仍不匹配或选项越界：回退 schema default，并记录 issue
  - `task_order`：过滤非字符串、未知 key、重复项，并自动补齐缺失任务
  - 旧类名分组如 `Batch_Files.DicomToImage`：视为未知 group，忽略且记录 issue，不做自动迁移
- 任意一次成功写回用户配置后，输出的都是 schema 规范化后的完整 payload；未知/过时字段会被清理掉。

### 3. 设置页 warning 与恢复流程
- `SettingWindow` 顶部增加持久 warning card，来源于 `settings.get_config_health()`，不再只靠启动时一次性 toast。
- warning card 展示：
  - 哪个文件有问题：用户配置、仓库快照、或两者
  - 问题类型：解析失败、字段非法、旧 group、未知字段等
  - 当前处理结果：已回退到 schema 默认值继续运行
- warning card 始终提供“恢复默认配置”按钮；按钮行为固定为：
  - 弹出确认对话框
  - 确认后用 schema 生成的完整默认 payload 覆盖用户配置文件
  - 重新加载 `AppSettings`
  - 刷新设置页、任务顺序、任务设置导航和当前主题
  - 成功则清空相关 health issue，失败则保留 warning 并显示错误通知
- 仓库内 `configs/settings.json` 即使损坏，也不会被按钮改写；它只会触发 warning，不会被运行时自动修复。

### 4. 设置页和参数构建改为完全吃 schema
- 设置页不再根据“用户 JSON 里有没有某个 group”决定是否显示任务设置；改成根据 `TaskSpec.settings` 中是否存在 `visible=True` 的字段决定。
- `get_setting_entries('Batch_Files', group_name=task_key)` 改成按 schema 查询，默认值直接来自 schema，不依赖用户配置是否已有该组。
- 设置控件渲染规则固定为：
  - `options is not None` 且全为 bool：`SwitchSettingCard`
  - `options is not None`：`ComboBox`
  - 否则：`LineEdit`
- `build_task_params()` 改成：
  - 读取任务 schema 对应的 group values
  - 合并 `TaskSpec.default_params`
  - 再做构造签名校验
- 结果是：新增任务设置后，不需要再改 `AppSettings`、`SettingWindow` 或 `configs/settings.json` 结构代码；任务设置接入点只剩 `core/task_registry.py`。  
  说明：新增任务本身的图标元数据仍在 `ui/task_ui_registry.py`，这不在本次“设置系统单一真相源”范围内。

### 5. `configs/settings.json` 的角色调整
- `configs/settings.json` 保留为仓库内默认配置快照，但来源改成 schema 生成，不再手写维护为真相源。
- 新增一个明确的生成/校验入口，例如：
  - 运行时 helper：`build_default_settings_payload()`
  - 测试期断言：仓库里的 `configs/settings.json` 与 helper 生成结果完全一致
- 这样做的结果：
  - 新增任务设置时只改 schema
  - 仓库默认配置快照和运行时默认值不会漂移
  - 仓库快照损坏时，运行时仍可从 schema 恢复默认值

## Test Plan
- schema 生成测试：
  - `build_default_settings_payload()` 覆盖 General / Network / Task_Center / Batch_Files
  - 生成结果与仓库 `configs/settings.json` 完全一致
- `AppSettings` 加载测试：
  - 用户配置 JSON 损坏时，effective settings 回退到 schema 默认值，并记录 recoverable health issue
  - 仓库快照 JSON 损坏时，effective settings 仍能从 schema 生成，并记录 health issue
  - 旧类名任务分组会被忽略并记录 issue
  - 非法 `task_order` 仍按当前语义清洗并补齐
- 设置页测试：
  - warning card 在存在 health issue 时显示
  - 点击恢复按钮会弹确认框
  - 确认后重建用户配置并刷新页面
  - 任务设置导航根据 schema 显示，而不是根据用户 JSON 是否已有 group 显示
- 参数构建测试：
  - 任务 schema 默认值能传给 `build_task_params()`
  - `TaskSpec.default_params` 仍能补充非用户可配置默认值
  - 构造签名校验继续生效
- 回归测试：
  - 现有 GUI 任务顺序、设置入口、确认弹窗、CLI 参数拼装不回归
  - 新增一个 dummy `TaskSpec.settings` 后，不改 `AppSettings` 也能自动出现在设置页并进入参数构建

## Assumptions
- 本次统一 schema 覆盖 `Batch_Files`、`Task_Center`、`General`、`Network` 四类设置。
- `TaskSpec.default_params` 保留，但仅用于不暴露在设置页的运行时默认参数；用户可配置参数的默认值全部移到 `TaskSpec.settings`。
- “恢复默认配置”只覆盖用户配置文件，不改仓库内 `configs/settings.json`。
- 仓库内 `configs/settings.json` 继续保留为可读快照，但运行时默认值不再依赖它。
- 任务整体接入仍需任务实现和 GUI 图标元数据；本次收敛的是“任务设置接入点”，不是“任务所有接入点”。
