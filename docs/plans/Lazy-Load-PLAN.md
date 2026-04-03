# RD-tools 懒加载改造方案

## Summary

本次改造仍采用三条主线：

- 主线一：模块级懒加载，移除 `main.py` 对任务类的顶层直接导入。
- 主线二：对 `ECGHandler`、`GenSubtitles` 等重任务补做依赖级懒加载，只下沉明显重依赖，不引入运行时单例或模型缓存。
- 主线三：设置页与执行链路从“依赖任务类对象/类名”改为“依赖 descriptor 中的 `settings_group`”。

本次固定的产品口径：

- 配置结构不变，继续沿用旧分组名，如 `ECGHandler`、`GenSubtitles`。
- 设置修改仅影响下次运行，不再尝试实时推送到已创建或正在运行的任务实例。
- 每次执行都创建新的 binding 和新的 handler 实例。
- 同一个任务不允许并发；不同任务维持现有可并发能力。
- 任务加载或初始化失败时，走“任务日志 + 现有 `InfoBar` 错误通知”双通道。

## Key Changes

### 1. 任务元数据与类加载

- `TaskDescriptor` 改为纯元数据：`key`、`title`、`description`、`icon`、`module_path`、`class_name`、`settings_group`、`default_params`。
- `build_task_descriptors()` 不再引用任务类，只填充模块路径、类名和设置分组。
- 新增统一 `TaskLoader`，按 `(module_path, class_name)` 缓存类对象。
- `TaskLoader` 只缓存类，不缓存实例；缓存访问需要加锁，保证跨任务并发时安全。

### 2. 执行生命周期

- 放弃“每任务常驻 binding + 常驻 handler”的结构，改为“每次执行新建 one-shot binding”。
- UI 注册阶段只注册 descriptor 和启动回调，不预先构造 handler。
- 点击执行后顺序固定为：
  1. 校验工作目录和勾选目录
  2. 读取当前设置并生成确认弹窗内容
  3. 记录开始日志，并追加“正在加载任务实现...”提示
  4. 将该 `task_key` 标记为运行中
  5. 创建本次执行专属 binding 并启动线程
- 线程内顺序固定为：
  1. `TaskLoader` 加载类
  2. 用 `settings_group` 从 `AppSettings` 取参数
  3. 合并 `default_params`
  4. 实例化新的 handler
  5. 连接 `handler.result_signal`
  6. 设置工作目录并执行任务
  7. `finally` 中关闭日志会话并清理活动 binding

### 3. 构造阶段消息与异常补齐

- 增加“bootstrap reporter”机制，作用范围仅覆盖“实例化前后到信号连接前”这一小段启动期。
- `BatchFilesBinding` 在创建 handler 前安装 bootstrap reporter；`FilesBasic.send_message()` 在 bootstrap reporter 存在时，除保留现有 `_pending_log_messages` 外，还要立即把消息转发到当前任务日志。
- 这样可以保留构造阶段的参数修正告警和初始化提示，例如 `BiliVideos.__init__()`、`ECGHandler.__init__()` 中的 `send_message()`。
- handler 实例化成功后，再连接 `handler.result_signal`，随后移除 bootstrap reporter。
- 如果构造阶段抛异常：
  - 已经通过 bootstrap reporter 发出的告警要保留在任务日志中
  - binding 再补发一条最终失败消息
  - UI 弹出错误通知
  - 任务状态标记为失败
- 本期“日志”指页面任务日志；构造失败前不要求额外落盘文件日志。

### 4. 设置页与执行链路解耦

- 设置页不再通过 `descriptor.operation_cls.__name__` 判断分组，统一改用 `descriptor.settings_group`。
- `has_task_settings()`、任务设置导航、任务设置详情渲染、确认弹窗中的“当前设置”读取，全部基于 `settings_group`。
- 删除 `settings.changed_signal -> live handler.update_setting` 这条实时同步链路。
- `AppSettings.get_class_params()` 可继续沿用现有配置结构和取值规则，但调用方统一传入 `descriptor.settings_group`。
- “设置修改仅影响下次运行”是本次改造后的正式语义，不保留当前系统的实时推送行为。

### 5. 重模块依赖下沉

- 第一阶段仅处理已确认的重项：
  - `ECGHandler`：优先下沉 `matplotlib` 相关导入，以及其他启动期明显重的科学计算依赖
  - `GenSubtitles`：优先下沉 `faster_whisper / WhisperModel` 相关导入和初始化
- 本期不引入模型缓存、运行时单例或共享 backend。
- 若某个任务构造期仍保留轻量参数校验和告警，允许存在；但真正的重依赖初始化应从导入链和构造阶段移出。

## Threading Rules

- `TaskLoader` 只缓存类对象，类缓存可以跨线程复用，但必须是线程安全的。
- handler 实例绝不跨线程共享；每次执行一个新实例，且实例只在创建它的执行线程内使用和销毁。
- UI 启动后不再直接读取或写入 handler 属性；UI 与任务运行态的交互只通过：
  - descriptor 元数据
  - 任务日志信号
  - 运行状态信号
  - 完成/失败通知
- 同一 `task_key` 只允许一个活动 binding；不同 `task_key` 可保持并发执行。
- 任何“为了方便”把 handler 引回 UI 线程做状态读取、设置同步或缓存复用的做法，均不属于本方案允许范围。

## Test Plan

- 启动应用后，任务列表、设置页、帮助页都能正常打开，且任务注册不再依赖任务类顶层导入。
- 设置页中各任务的设置入口和设置项展示与当前版本一致，仍按旧分组名映射。
- 修改设置后，不影响正在运行的任务；重新执行同一任务时使用新设置。
- `ECGHandler`、`GenSubtitles` 在启动阶段不再被提前导入，首次执行时按需加载。
- `BiliVideos`、`ECGHandler` 构造阶段产生的参数修正告警能进入页面任务日志。
- `GenSubtitles` 构造阶段抛出依赖异常时，任务日志中能看到启动上下文和最终失败信息，页面出现错误通知。
- 同一任务运行中再次点击不会启动第二个实例；不同任务仍可分别执行。
- 故意制造错误的 `module_path`、`class_name` 或实例化异常时：
  - 任务状态变为失败
  - 页面任务日志保留启动期信息和最终错误
  - 页面出现错误通知

## Assumptions

- `settings_group` 与当前配置中的旧类名一一对应，并继续作为设置分组标识。
- 本期不改 `settings.json` 结构，不做配置迁移。
- 本期不处理打包发布问题。
- 本期不引入任务实例缓存、模型缓存、DI、插件系统或 worker 进程隔离。
- 跨任务并发保持现状；如果后续要限制全局并发，需要单独立项。
