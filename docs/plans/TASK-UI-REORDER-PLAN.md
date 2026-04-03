## 任务列表可拖拽排序定稿

### Summary
- 新增“任务显示顺序”能力，只影响展示顺序，不改任务 `key`、标题、功能、设置分组。
- 顺序配置存到用户级 `settings.json` 的 `Task_Center.task_order`，配置字段通过 `AppSettings` 暴露。
- 同一份顺序同时作用于“任务中心”和“设置 > 任务设置”；设置页只跟随顺序，不提供拖拽入口。
- 默认顺序不再依赖 `build_task_descriptors()` 当前声明顺序，而是由 `AppSettings` 内置默认值决定。

### Key Changes
- 在 `AppSettings` 中新增：
  - `Task_Center_Settingmap = {"task_order": ("Task_Center", "task_order")}`
  - 代码级默认顺序常量 `DEFAULT_TASK_ORDER`
  - `get_task_order(available_task_keys: list[str]) -> list[str]`
  - `save_task_order(task_keys: list[str]) -> bool`
  - 启动期一次性 warning 缓冲，例如 `consume_startup_warnings() -> list[str]`
- `DEFAULT_TASK_ORDER` 固定为：
  - `files-renamer`
  - `bilibili-export`
  - `subtitle-generation`
  - `mac-cleaner`
  - `merge-colors`
  - `split-colors`
  - `twist-images`
  - `ecg-handler`
  - `dicom-processing`
- `configs/settings.json` 增加默认节点：
  - `"Task_Center": { "task_order": [...] }`
  - 内容与 `DEFAULT_TASK_ORDER` 保持一致；新建用户配置时会直接带上该节点。
- 任务顺序解析规则固定为：
  - `Task_Center` 或 `task_order` 缺失：静默回退到 `DEFAULT_TASK_ORDER`
  - `task_order` 不是列表：整项视为非法，回退默认顺序，并记录一次 warning
  - 列表里有非字符串、未知 key、重复 key：过滤并去重，保留第一次出现，同时记录一次 warning
  - 配置里未覆盖到的任务，按 `build_task_descriptors()` 的原始声明顺序自动追加到末尾
  - 启动时不自动改写非法配置；只有用户拖拽成功后才写回修正后的顺序
- 运行时数据流固定为：
  - `main.py` 先构建 descriptor 列表，再通过 `settings.get_task_order()` 得到排序后的 descriptor 列表
  - `MainWindow` 持有当前唯一的有序 descriptor 列表，创建 `FileWindow` / `SettingWindow` 时都使用它
  - `FileWindow` 左侧任务列表开启 `InternalMove` 拖拽排序，交互为“按住后直接拖拽”，不做长按延时
  - 拖拽落下后，`FileWindow` 发出新的 `task_key` 顺序给 `MainWindow`
  - `MainWindow` 立即调用 `save_task_order()` 持久化；成功后同步刷新 `SettingWindow` 的任务导航顺序
- UI 行为固定为：
  - “任务中心”左侧列表支持拖拽排序
  - “设置 > 任务设置”左侧列表不支持拖拽，只按同一份顺序展示
  - 设置页仍只显示“有独立设置项”的任务；实现方式是先按全局顺序排序，再过滤无设置项任务
  - 拖拽后尽量保留当前选中任务；若当前任务仍存在，则继续选中它
  - 若写回 `settings.json` 失败，撤销本次界面排序变化，并弹错误提示
- warning 反馈链路固定为：
  - 非法 `task_order` 在每次应用启动时最多提示一次 `InfoBar.warning`
  - 同时向控制台打印一条 warning
  - 缺失配置不提示

### Test Plan
- 单元测试覆盖：
  - 缺失 `Task_Center`
  - 缺失 `task_order`
  - `task_order` 为非列表
  - 列表中含未知 key、重复 key、非字符串
  - 列表只写部分任务时，新任务自动追加到末尾
  - 默认顺序严格等于约定的 9 个 key 顺序
- 手工验证覆盖：
  - 任务中心拖拽后立即生效并写回用户级 `settings.json`
  - 重启应用后顺序保持
  - 设置页“任务设置”顺序与任务中心一致，但仍只显示有设置项的任务
  - 拖拽不影响任务执行、日志、当前选中状态和“修改设置”入口
  - 非法配置启动时只弹一次 warning，且控制台有对应日志
  - 写回失败时排序回滚且有错误提示

### Assumptions
- “日志台打印 warning”解释为控制台 `print`/日志输出，不写入任务运行日志。
- 排序保存时机固定为“拖拽落下后立即写回”，不延迟到退出应用。
- 不额外做历史配置迁移或兼容脚本；现有旧配置通过“运行时回退 + 下一次成功拖拽后写回”自然收敛。
