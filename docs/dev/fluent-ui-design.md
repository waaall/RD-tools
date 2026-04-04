# RD-tools Fluent UI 设计文档

本文档描述 RD-tools 当前基于 `PySide6-Fluent-Widgets` 的 UI 设计思路、代码结构、主题机制、页面组织方式，以及后续扩展时应遵守的实现规则。

- 为什么现在的 UI 这样设计
- 哪些层可以改，哪些层不要轻易动
- 如何新增任务、设置项、页面和主题样式
- 如何在保持模块化和可读性的前提下继续扩展

## 1. 设计目标

当前 UI 体系的核心目标是四个：

1. 统一
   - 避免每个页面自己决定颜色、间距、按钮样式和信息层级。
   - 避免局部 `setStyleSheet()` 到处散落，导致后续维护成本越来越高。

2. 模块化
   - 业务模块继续独立存在。
   - UI 层负责“展示、交互、状态反馈”，不去吞掉业务逻辑。
   - 配置层继续由现有 `AppSettings` 负责，不强行重构为另一套配置系统。

3. 可扩展
   - 新增任务时，不需要重新发明一套页面结构。
   - 新增设置项时，不需要重新手写一套表单布局。
   - 后续新增更多页签或页面时，可以沿用同一套窗口壳和主题机制。

4. 现代化
   - 使用 `QFluentWidgets` 提供统一的导航、卡片、设置卡片、分段切换、通知等组件。
   - 用统一的视觉 token 和布局规则代替旧版的灰底白控件拼接式界面。

## 2. 设计原则

### 2.1 表现层和业务层分离

本次改造遵循“重做表现层，不重写业务层”的原则。

保留不变的部分：

- 各个批处理模块类
- 线程执行模型 `BatchFilesBinding`
- `settings.json` 的结构
- `AppSettings` 的读写机制和信号机制

重构的部分：

- 主窗口壳
- 页面骨架
- 任务中心的组织方式
- 设置页的控件表现
- 帮助页的阅读方式
- 通知和状态反馈方式
- 统一主题和 QSS 管理

### 2.2 约束优先于自由发挥

后续 UI 开发不鼓励“看哪个页面需要就单独写一点样式”。

推荐顺序是：

1. 先确认是否已有组件或模式可复用
2. 再决定是否在统一 QSS 中补一个规则
3. 最后才考虑页面内局部样式

默认不允许：

- 页面内硬编码主题色
- 同一类控件使用多套高度和间距
- 同一页面混用过多布局风格
- 为了快速修一个问题，直接在页面里堆 `setStyleSheet()`

### 2.3 页面是装配层，不是业务层

页面类的职责应保持在以下范围：

- 创建布局
- 接收用户输入
- 切换显示状态
- 展示日志和通知
- 调用业务层提供的接口

页面不应承担：

- 复杂数据处理
- 文件扫描算法
- 参数推导逻辑
- 配置文件底层读写逻辑

## 3. 当前 UI 架构

### 3.1 目录结构

当前 UI 相关代码主要分布如下：

- `main.py`
  - 应用入口
  - 任务描述注册
  - 线程绑定与任务执行接线
- `main_window.py`
  - Fluent 主窗口壳
  - 顶层导航
  - 顶层通知分发
- `widgets/file_page.py`
  - 任务中心页面
- `widgets/setting_page.py`
  - 设置页
- `widgets/help_page.py`
  - 帮助页
- `ui/task_descriptor.py`
  - 任务元数据结构
- `ui/theme.py`
  - 主题入口与应用级样式加载
- `ui/qss/light/app.qss`
  - 明亮主题的应用级样式
- `ui/qss/dark/app.qss`
  - 暗色主题的应用级样式

### 3.2 分层关系

推荐把当前 UI 理解为四层：

1. 应用层
   - `main.py`
   - 负责启动应用、注册任务、连接线程和页面

2. 窗口壳层
   - `main_window.py`
   - 负责导航、主题应用、顶层通知、页面挂载

3. 页面层
   - `widgets/file_page.py`
   - `widgets/setting_page.py`
   - `widgets/help_page.py`
   - 负责布局和交互

4. 视觉系统层
   - `ui/theme.py`
   - `ui/qss/*`
   - 负责主题、颜色、应用级样式

## 4. 依赖与版本策略

当前 UI 体系依赖：

```txt
PySide6-Fluent-Widgets==1.11.1
```

安装方式：

```bash
python -m pip install PySide6-Fluent-Widgets==1.11.1 -i https://pypi.org/simple
```

注意事项：

- 不要同时安装 `PyQt-Fluent-Widgets`、`PyQt6-Fluent-Widgets`、`PySide2-Fluent-Widgets` 和 `PySide6-Fluent-Widgets`。
- 这些包共享 `qfluentwidgets` 包名，会互相冲突。
- 本项目默认只使用公开版，不依赖 Pro 组件，也不默认使用 `[full]` 版本。

## 5. 主题系统设计

### 5.1 主题入口

主题逻辑集中在 `ui/theme.py`，而 `Auto` 模式下的同步调度放在 `main_window.py`。

当前入口函数：

- `resolve_theme(theme_name)`
- `is_auto_theme(theme_name)`
- `detect_system_theme()`
- `apply_app_theme(theme_name, app)`
- `refresh_auto_app_theme(app)`

其职责是：

- 把现有配置里的 `theme` 映射到 `Theme.LIGHT`、`Theme.DARK` 或 `Theme.AUTO`
- 调用 `setTheme()` 应用明暗主题
- 调用 `setThemeColor()` 统一设置主强调色
- 按当前主题加载对应的应用级 QSS
- 在 `Auto` 模式下刷新 `qconfig.theme` 并补发主题变更信号，保证 Fluent 组件和应用级 QSS 一起更新

当前 `Auto` 模式的设计约束：

- 不再使用 `QFluentWidgets` 自带的 `SystemThemeListener(QThread)`
- 只在 `theme == Auto` 时启用主线程 `QTimer`
- 轮询间隔由 `SYSTEM_THEME_SYNC_INTERVAL_MS` 控制，当前为 `1000 ms`
- 退出时只停止定时器，不在 `closeEvent()` 中等待后台主题线程退出

这样做的原因：

- 避免关闭窗口时固定等待主题监听线程
- 避免 Windows 上出现 `QThread: Destroyed while thread '' is still running`
- 保留 `Auto` 模式下跟随系统明暗主题的能力

### 5.2 当前视觉 token

当前设计使用的核心 token：

- Fluent 组件主题色：`#29526f`
- 应用级 accent token：`#2F6FED`
- 成功色：`#2AA676`
- 警告色：`#D49C1D`
- 错误色：`#C55252`
- 暗色背景：`#202124`
- 暗色面板：`#26282C`
- 暗色卡片：`#2C2F33`
- 主文本：`#F5F7FA`
- 次级文本：`#B8C0CC`

这些 token 不是严格意义上的设计 token 系统，但目前已经足够支撑应用级统一风格。

### 5.3 布局 token

当前通用布局规则：

- 页面外边距：`24`
- 卡片内边距：`16`
- 区块间距：`16`
- 行内控件间距：`12`
- 控件目标高度：`36`
- 卡片圆角：`10`

后续新增页面时，尽量沿用这些数值，不要每页各写一套。

### 5.4 QSS 策略

QSS 目前只用于“应用级规则”，主要解决：

- 页面标题风格
- 次级文案风格
- 卡片边框与背景
- 列表选中态和 hover 态
- 日志区与文档区外观
- 状态文案颜色

推荐做法：

- 全局风格改动优先改 `ui/qss/light/app.qss` 和 `ui/qss/dark/app.qss`
- 不要在页面里直接写一堆颜色值
- 必须动态变化的局部样式，才在页面里用属性或少量逻辑处理

## 6. 主窗口壳设计

### 6.1 选型

主窗口使用 `FluentWindow`，而不是原来的：

- `QMainWindow`
- `QDockWidget`
- `QStackedWidget`

这样做的原因：

- 导航结构更稳定
- 左侧导航符合工具型桌面应用模式
- 页面切换和整体视觉层次统一
- 减少手工维护 Dock 和栈切换逻辑

### 6.2 当前导航结构

当前导航固定为三项：

- 顶部：`任务中心`
- 顶部：`设置`
- 底部：`帮助`

约束：

- 默认首页是 `任务中心`
- 页面对象在挂入 Fluent 导航前必须设置 `objectName`
- 以后新增一级页面，应继续通过 `addSubInterface()` 注册，不要自己绕开导航栈单独 show 新窗口，除非确实是对话框场景

### 6.3 顶层通知

顶层通知统一由 `MainWindow.show_notification()` 分发。

页面内部不直接依赖旧 `statusBar()`，而是通过信号把消息交给主窗口，主窗口再调用：

- `InfoBar.success`
- `InfoBar.error`
- `InfoBar.warning`
- `InfoBar.info`

这样做的好处：

- 通知风格统一
- 页面层不直接关心通知组件细节
- 后续如果替换通知位置或持续时间，只需要改一处

## 7. 任务中心设计

### 7.1 设计目标

任务中心从“长卡片堆叠”改成“左列表 + 右详情”的结构，核心目的是：

- 让任务数量增长时页面仍然稳定
- 让工作目录和处理范围只维护一份
- 让每个任务的日志彼此独立但展示方式统一
- 让执行按钮、状态和日志位置固定，减少认知负担

### 7.2 页面结构

当前页面分为三层：

1. 页头
   - 标题
   - 说明文案
   - 当前工作目录摘要

2. 左侧任务列表
   - 展示当前所有任务
   - 当前任务高亮
   - 运行中的任务会带“运行中”状态文本

3. 右侧任务工作区
   - 顶部任务概览卡片
   - 底部分段内容卡片

### 7.3 工作目录模型

工作目录是全局状态，保存在 `FileWindow` 中：

- `_work_folder`
- `_work_folder_items`

目录选择只保留一份，不再和任务卡片绑定。

所有任务执行前都复用：

- 当前工作目录
- 当前勾选的一级子目录

这使任务中心更像“统一工作台”，而不是一组互相独立的小面板。

### 7.4 任务元数据模型

任务列表不再使用匿名元组，而是统一使用 `TaskSpec + TaskDescriptor` 两层模型：

```python
@dataclass(frozen=True, slots=True)
class TaskSpec:
    key: str
    title: str
    description: str
    module_path: str
    class_name: str
    default_params: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TaskDescriptor:
    task_spec: TaskSpec
    icon: Any
```

字段职责：

- `key`
  - 稳定的任务标识，也是配置分组名
- `title`
  - 同时供 GUI 和 CLI 展示的标题
- `description`
  - GUI 详情区和 CLI 提示都可复用的说明文案
- `module_path` / `class_name`
  - 运行时懒加载任务类所需的最小信息
- `icon`
  - GUI 展示元数据，不进入核心 registry
- `default_params`
  - 默认参数补丁

### 7.5 任务注册流程

任务注册真相源不再放在 `main.py`，而是拆成：

- `core/task_registry.py`
- `ui/task_ui_registry.py`

流程如下：

1. 从 `core.task_registry` 读取 `TaskSpec`
2. 从 `ui.task_ui_registry` 补齐图标并生成 `TaskDescriptor`
3. 调用 `register_batch_operation()`
4. 运行时用 `TaskLoader` 懒加载业务处理类
5. 通过 `core.task_params.build_task_params()` 取配置并补齐默认参数
6. 实例化业务处理类
7. 用 `BatchFilesBinding` 包一层线程执行逻辑
8. 把信号接到 `FileWindow`
9. 把任务注册到左侧任务列表

这样做的优点是：

- UI 和业务处理类通过 descriptor 解耦
- GUI 和 CLI 可以共享同一份核心任务注册与参数构建逻辑
- 后续新增任务时，不需要改页面布局逻辑
- 同一任务的说明、图标、执行行为集中管理

### 7.6 任务状态与日志

每个任务在 UI 层维护以下运行态：

- `_operation_logs`
- `_task_states`
- `_running_tasks`

设计意图：

- 日志和任务一一对应
- 切换任务不会丢日志
- 是否运行中是 UI 自己的状态，不反向污染业务模块

通知策略：

- 阻断性问题：`InfoBar.error`
- 成功完成：`InfoBar.success`
- 普通过程信息：只写入日志

## 8. 设置页设计

### 8.1 为什么不迁移到 QConfig

`QFluentWidgets` 自带 `QConfig` 和配置卡片体系，但当前项目保留 `AppSettings`，不迁移的原因是：

- 现有 JSON 结构已经在使用
- 业务模块依赖 `AppSettings.changed_signal`
- 当前 UI 改造的目标是重构表现层，不是重构配置系统

因此当前策略是：

- 使用 Fluent 风格卡片作为展示组件
- 继续由 `AppSettings` 负责配置读写和信号广播

### 8.2 设置页结构

设置页固定结构：

- 页头标题与说明
- `ScrollArea`
- 多个 `SettingCardGroup`

分组来源是 `AppSettings.get_main_categories()` 返回的一级分类。

### 8.3 当前控件映射规则

设置项显示规则如下：

- 预定义选项为布尔值：`SwitchSettingCard`
- 预定义选项为普通枚举：自定义 `AppComboBoxSettingCard`
- 普通文本：自定义 `AppLineEditSettingCard`
- 包含 `api_key` 的字段：密码输入模式

注意：

- 当前没有直接使用 `ComboBoxSettingCard`，因为该组件设计上依赖 `QConfig` 的 `OptionsConfigItem`
- 因此这里使用“继承 `SettingCard` 的轻量适配卡片”来保持当前配置系统不变

### 8.4 保存策略

当前设置页采用“即时保存”：

- 值变化后立即调用 `save_settings()`
- 成功时静默
- 失败时弹 `InfoBar.error`

这个策略适合当前工具型软件，因为：

- 配置项数量有限
- 用户期望是改完立即生效
- 无需再增加额外的“保存”按钮和状态管理

### 8.5 增加设置项的方法

如果要新增设置项，建议按以下顺序：

1. 在 schema 定义里新增字段：
   任务参数放到 `core/task_registry.py` 的 `TaskSpec.settings`
   通用参数放到 `core/settings_schema.py`
2. 确保字段默认值、类型、选项和业务类接收参数一致
3. 打开设置页验证该字段是否自动生成
4. 运行 schema 快照测试，确认 `configs/settings.json` 与生成结果一致

如果新增的是一种新的输入类型，而不是布尔/枚举/文本之一，则再考虑新增新的卡片适配组件。

## 9. 帮助页设计

帮助页保持最小复杂度。

原则：

- 继续复用已有 markdown 文档
- 不在帮助页里增加复杂解析逻辑
- 阅读体验统一，但不把它改造成文档系统

当前结构：

- 页头
- `SegmentedWidget`
- 文档卡片
- `QTextBrowser`

页签固定：

- 用户文档
- 开发文档

后续如需增加更多文档类型，优先扩展 `SegmentedWidget + stacked widget`，而不是新增多个独立窗口。

## 10. 通知与状态反馈策略

当前 UI 明确区分三类反馈：

1. 全局即时提示
   - 用 `InfoBar`
   - 适用于错误、成功、阻断提示

2. 页面内状态文本
   - 适用于工作目录摘要、选择状态、任务状态
   - 这些状态需要长期留在页面上，而不是弹一下就消失

3. 日志流
   - 适用于处理过程消息
   - 只记录在当前任务的日志面板中

不要把“所有消息都弹通知”，也不要把“所有反馈都塞日志”。这两者都不利于长期使用。

## 11. 扩展方法

### 11.1 如何新增一个任务

推荐步骤：

1. 新增或准备好对应的业务处理类
2. 在 `main.py` 的 `build_task_descriptors()` 中增加一个 `TaskDescriptor`
3. 确保 `operation_cls` 可由现有参数体系初始化
4. 如果需要默认参数，在 `default_params` 中补充
5. 运行应用，验证：
   - 左侧任务是否出现
   - 描述是否正确显示
   - 日志是否独立
   - 运行态是否正确

示例：

```python
TaskDescriptor(
    key='new-task',
    title='新任务',
    description='这里写任务说明。',
    icon=FIF.DOCUMENT,
    operation_cls=MyTask,
    default_params={'parallel': False},
)
```

### 11.2 如何新增一个一级页面

如果未来要新增新的一级导航页：

1. 新建页面类
2. 给页面设置 `objectName`
3. 在 `MainWindow._init_interfaces()` 中用 `addSubInterface()` 注册
4. 按需要接入页面自己的通知信号

不建议：

- 直接在 `main.py` 随手 `show()` 一个新 `QWidget`
- 让一级页面绕过 Fluent 导航栈存在

### 11.3 如何新增一条应用级样式

新增全局样式时：

1. 优先添加对象名或属性
2. 再到 `ui/qss/light/app.qss` 和 `ui/qss/dark/app.qss` 中补规则
3. 明暗主题规则尽量成对出现

例如：

- 给某个组件设置 `setObjectName('MyWidget')`
- 在明暗 QSS 中分别写 `QWidget#MyWidget` 的规则

### 11.4 如何修改主题色

主题色目前由 `ui/theme.py` 中的 `APP_THEME_COLOR` 控制。

如果未来需要支持用户自定义强调色，建议：

1. 先在配置中增加字段
2. 再让 `apply_app_theme()` 从配置读取
3. 最后评估是否需要补更多色阶 token

不要只改 QSS 里的局部蓝色值，这会导致组件主题色和应用级样式不同步。

## 12. 开发约束和建议

### 12.1 不建议做的事

- 在页面里继续大量写 `setStyleSheet()`
- 为每个设置项单独拼一套布局
- 在主窗口之外维护另一套页面切换逻辑
- 在 `modules/__init__.py` 中恢复全量重模块导入
- 让页面直接依赖底层配置文件路径和 JSON 细节

### 12.2 推荐做的事

- 保持页面类尽量薄
- 让 `TaskDescriptor` 继续作为任务元数据中心
- 让全局视觉规则尽量回到 `ui/qss`
- 新增 UI 状态时先考虑是否属于“通知 / 状态文本 / 日志流”中的哪一种
- 继续把业务逻辑留在 `modules` 层

## 13. 已知限制

### 13.1 FluentWindow 的无界面环境兼容性

当前在 `QT_QPA_PLATFORM=offscreen` 的 smoke test 下，最小化 `FluentWindow()` 本身就会直接退出。这说明：

- 这不是当前项目特有的问题
- 更可能是 `QFluentWidgets` 或 `qframelesswindow` 在无界面平台插件下的限制

因此当前测试策略是：

- 页面类做 `offscreen` 实例化验证
- 真实窗口壳的验证放到桌面环境中执行

### 13.2 视觉风格偏 Fluent / WinUI

虽然项目运行在 macOS 上也没有问题，但 `QFluentWidgets` 的视觉语言仍然更偏 Windows Fluent 风格。

这意味着：

- 它更统一、更现代
- 但不追求平台原生 macOS 风格

这是当前设计上的主动取舍。

### 13.3 设置系统仍是桥接模式

设置页目前使用的是“Fluent 卡片外观 + 现有 `AppSettings` 读写逻辑”的桥接方案。

优点是迁移风险低。

代价是：

- 不能直接复用所有基于 `QConfig` 的高级设置卡片能力
- 自定义字段类型需要自己补适配卡片

## 14. 推荐测试清单

每次改动 UI 后，至少检查以下内容：

1. 启动应用，确认主窗口能打开
2. 导航切换正常
3. 任务中心选择目录、勾选目录、切换任务、清空日志正常
4. 任务运行时状态变化正常
5. 设置页所有卡片可正常显示和写回
6. `Light`、`Dark`、`Auto` 三种主题模式都能正常工作
7. `Auto` 模式下切换系统明暗主题后，界面可在轮询周期内自动同步
8. 关闭窗口时不再因主题同步出现明显额外等待或 `QThread` 销毁警告
9. 帮助页文档能加载、页签能切换、外链可打开
10. 在 `1280x720` 下没有明显重叠和跑版

## 15. 后续演进建议

后续如果继续提升 UI，建议优先级如下：

1. 打磨任务列表的信息密度和选中态细节
2. 统一更多图标语义，减少图标临时选择
3. 补一套更完整的组件命名和对象名规则
4. 抽出更正式的设计 token 层，而不是只依赖当前常量和 QSS
5. 视需要评估是否把设置系统迁移到 `QConfig`

在那之前，当前这套实现已经足够作为稳定的 Fluent UI 基线。
