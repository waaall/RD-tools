# RD-tools 懒加载改造最小实施方案

## Summary

本次改造采用三条主线并行推进：

- 以模块级懒加载为主线，移除 `main.py` 对各任务模块的顶层直接导入。
- 对 `ECGHandler`、`GenSubtitles` 等重任务补充依赖级懒加载，仅下沉明显重依赖，不引入运行时单例或模型缓存。
- 将设置页和执行链路从“依赖任务类对象/类名”改为“依赖任务元数据中的 `settings_group`”。

本次方案保持以下约束不变：

- 配置文件结构不变。
- 旧的设置分组名继续沿用当前类名，例如 `ECGHandler`、`GenSubtitles`。
- 设置修改仅影响下次运行。（当前系统会尝试把设置实时推给正在运行的任务实例。这是要修改的后的结果）
- 同一个任务不允许并发多次运行。
- 任务加载失败时，使用现有通知机制弹出错误，同时写入该任务日志。

## Key Changes

### 1. 任务元数据与加载方式

- 将 `TaskDescriptor` 从“持有 `operation_cls`”改为“只持有元数据”。
- `TaskDescriptor` 定稿字段为：`key`、`title`、`description`、`icon`、`module_path`、`class_name`、`settings_group`、`default_params`。
- `settings_group` 直接沿用当前配置中的旧分组名，不做迁移。
- `build_task_descriptors()` 改为填充模块路径和类名字符串，不再导入任务类。
- 新增统一的 `TaskLoader`，按 `(module_path, class_name)` 缓存类对象；只缓存类，不缓存实例。

### 2. 执行生命周期改造

- 放弃“每任务一个常驻 binding + 常驻 handler”的方式，改为“每次执行新建一个 binding 和一个 handler 实例”。
- `BatchFilesBinding` 不再在构造时接收 `handler_object`；改为接收 `descriptor`、`settings`、`file_window` 等轻量上下文。
- 点击执行后，先完成目录校验和确认弹窗；确认通过后立即：
  - 记录开始日志
  - 将该 `task_key` 标记为运行中
  - 创建本次执行专属的 `BatchFilesBinding`
  - 将其放入活动绑定容器，直到执行结束后清理
- 类加载和实例化统一放到执行线程中完成，避免首次点击重任务时阻塞 UI 线程。
- 线程中的执行顺序固定为：
  1. 通过 `TaskLoader` 加载类
  2. 通过 `settings_group` 从 `AppSettings` 取参数
  3. 合并 `default_params`
  4. 创建新的 handler 实例
  5. 连接 `handler.result_signal`
  6. 设置工作目录并执行任务
  7. `finally` 中关闭日志会话
- 同一任务再次点击时，如果该 `task_key` 已在运行中，则直接拒绝第二次启动，保持当前“阻止并发”的行为。

### 3. 设置页与确认弹窗解耦

- 设置页不再通过 `descriptor.operation_cls.__name__` 判断设置分组，统一改用 `descriptor.settings_group`。
- `SettingWindow.has_task_settings()`、任务设置导航生成、任务设置内容渲染，全部改为基于 `settings_group`。
- 执行确认弹窗中的“当前设置”列表也改为基于 `settings_group` 读取。
- 删除运行期的 `settings.changed_signal -> handler.update_setting` 实时同步链路。
- 保留 `AppSettings.get_class_params()` 现有配置结构和取值规则，只是调用方改为传入 `descriptor.settings_group`。

### 4. 重模块依赖下沉

- 第一阶段仅处理明显重依赖，不引入任务级单例或共享运行时。
- `ECGHandler` 优先下沉 `matplotlib` 相关导入，以及其他确认为启动期重项的依赖。
- `GenSubtitles` 优先下沉 `faster_whisper / WhisperModel` 相关导入和初始化。
- 目标是削减启动导入链，不追求本阶段解决“重复进入任务时的二次初始化成本”。

### 5. 错误反馈与状态语义

- 任务模块加载失败、类不存在、实例化失败，都视为“执行前失败”。
- 失败时固定执行以下动作：
  - 向该任务日志追加错误消息
  - 将任务状态标记为“上次运行失败”
  - 通过现有 `InfoBar` 通知链路弹出错误提示
  - 清理本次活动 binding 和运行中标记
- “弹窗”在本方案中定义为沿用当前 `MainWindow.show_notification()` 的 `InfoBar` 机制，不新增模态对话框。
- 为避免用户误判，执行启动后应尽早写入一条类似“正在加载任务实现...”的日志，覆盖类加载阶段的等待感。

## Test Plan

- 启动应用后，任务列表、设置页、帮助页都能正常打开，且不再依赖任务模块的顶层导入完成注册。
- 设置页中各任务的设置入口、设置项展示与当前版本保持一致，分组仍按旧类名映射。
- 修改设置后，不影响正在运行的任务；下次重新执行时使用新设置。
- 轻量任务可正常执行，日志、状态和完成提示保持现有行为。
- `ECGHandler`、`GenSubtitles` 首次执行时按需加载，启动阶段不再提前导入。
- 同一任务运行中再次点击，不会产生第二个执行实例。
- 故意制造错误的 `module_path`、`class_name` 或导入异常时：
  - 当前任务日志能看到错误
  - 任务状态变为失败
  - 页面出现错误通知
- 多个不同任务仍可按现有设计分别执行，不要求本次改造改变跨任务并发语义。

## Assumptions

- `settings_group` 与当前配置中的旧类名一一对应，并继续作为设置分组标识使用。
- 本期不改 `settings.json` 结构，不做配置迁移。
- 本期不处理打包发布问题。
- 本期不引入任务实例缓存、模型缓存、DI 容器、插件系统或 worker 进程隔离。
- 本期保持现有 UI 交互模式，仅调整任务加载、实例创建和设置解耦路径。
