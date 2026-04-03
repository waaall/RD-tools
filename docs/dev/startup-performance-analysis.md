# 应用启动耗时分析指南

本文档说明如何对 Python 桌面应用的启动耗时进行定位与分析。文档重点在于建立一致的分析方法，帮助开发人员回答“时间消耗在哪里”“应优先优化什么”这两个核心问题，而不是直接预设某一种优化方案。

正文以通用方法为主，最后附带一节 RD-tools 的示例说明，便于将方法与实际项目对应起来。

## 1. 启动耗时分析要回答的问题

在开始测量之前，应先明确启动耗时分析的目标。一次有效的分析，至少需要回答以下问题：

- 启动总耗时是多少。
- 耗时主要集中在哪个阶段，例如入口导入、依赖导入、配置初始化、UI 构造、首次任务注册，或打包产物的额外开销。
- 问题是否稳定可复现，是否只出现在冷启动、首次启动或特定环境下。
- 源码运行与打包运行的表现是否一致。
- 优化优先级应放在代码结构、依赖管理、缓存目录、运行环境，还是打包方式。

如果上述问题尚未回答，通常不建议直接进入代码优化阶段。否则很容易在非关键路径上投入过多精力。

## 2. 最小排查顺序

为避免分析过程失焦，建议按以下顺序开展启动耗时排查。

### 2.1 明确测量口径

首先定义“启动完成”的判定标准。常见口径包括：

- 主进程启动到主窗口首次显示。
- 主窗口显示到可交互。
- 命令发出到首页数据加载完成。

不同口径会直接影响结论。开发文档、测试记录和优化对比应使用同一口径。

### 2.2 先建立总耗时基线

在分解问题之前，应先获得整体启动时间，例如：

- 冷启动耗时。
- 热启动耗时。
- 源码运行耗时。
- 打包运行耗时。

这一步的目的不是定位根因，而是确认问题是否真实存在，以及后续优化是否有效。

### 2.3 按阶段拆分启动路径

启动耗时通常可以拆为以下几个部分：

- 入口模块导入。
- 业务模块导入。
- 配置、缓存、模型或资源初始化。
- GUI 应用对象创建。
- 主窗口及页面装配。
- 启动后立即执行的扫描、注册或预加载逻辑。

如果不先做阶段拆分，后续即使拿到了多个耗时数字，也很难判断它们之间的因果关系。

### 2.4 单独验证可疑模块或初始化步骤

当总耗时和分段耗时已经显示某一阶段明显偏重时，应继续拆到具体模块或具体初始化动作，例如：

- 某个重量级依赖是否在模块顶层导入。
- 某个对象的构造函数是否执行了大量 I/O。
- 某个页面是否在创建时立即加载数据或扫描文件。

### 2.5 最后再判断优化优先级

完成定位之后，再决定优化顺序。通常优先级如下：

- 减少启动路径上的不必要工作。
- 延迟加载重量级依赖。
- 将重初始化从“启动时”移动到“真正使用时”。
- 处理环境与缓存问题。
- 重新评估打包方式。

## 3. 常用分析手段

本节提供常见、可迁移的分析方法。示例命令中的模块名和函数名需要根据具体项目替换。

### 3.1 使用 `importtime` 分析导入链

`importtime` 适用于判断 Python 启动链路中哪些模块导入成本较高，尤其适合排查“程序尚未进入业务逻辑，启动时间已经明显偏长”的情况。

建议先定义一个简单的辅助函数：

```bash
profile_importtime() {
  local module="${1:-main}"
  local out="${2:-/tmp/${module//./_}_importtime_$(date +%Y%m%d_%H%M%S).txt}"

  python3 -X importtime -c "import ${module}" 2> "${out}" || return $?

  echo "importtime 日志: ${out}"
  echo
  tail -n 120 "${out}"
}
```

用法示例：

```bash
profile_importtime main
profile_importtime your_ui_module
profile_importtime your_feature_module
```

说明：

- `python3 -X importtime` 用于输出导入耗时统计。
- 输出会写入标准错误，因此需要使用 `2>` 重定向。
- `tail -n 120` 只适合快速预览，不适合作为正式结论。

如果需要按累计耗时排序，可配合以下汇总脚本：

```bash
summarize_importtime() {
  local out="$1"

  python3 - "$out" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = []

for line in path.read_text().splitlines():
    m = re.match(r'import time:\s+(\d+) \|\s+(\d+) \|\s+(.+)', line)
    if not m:
        continue
    self_us = int(m.group(1))
    cum_us = int(m.group(2))
    module = m.group(3).strip()
    rows.append((cum_us, self_us, module))

for cum_us, self_us, module in sorted(rows, reverse=True)[:30]:
    print(f'{cum_us/1000:8.1f} ms cum | {self_us/1000:8.1f} ms self | {module}')
PY
}
```

建议使用方式：

1. 先运行 `profile_importtime <入口模块>`，保留完整日志。
2. 再运行 `summarize_importtime <日志文件>`，按累计耗时查看最重模块。
3. 对前几名模块做进一步验证，而不是仅凭日志尾部做判断。

### 3.2 使用分段计时建立阶段视图

如果目标是判断“慢在导入、初始化还是 UI 构造”，分段计时通常比单次总耗时更有价值。

入口模块导入示例：

```bash
python3 - <<'PY'
import time

start = time.perf_counter()
import your_entry_module
print(f'import your_entry_module: {time.perf_counter() - start:.3f}s')
PY
```

主窗口模块导入示例：

```bash
python3 - <<'PY'
import time

start = time.perf_counter()
import your_ui_module
print(f'import your_ui_module: {time.perf_counter() - start:.3f}s')
PY
```

如果项目已经具备明确的初始化函数，也可以进一步分段：

```bash
python3 - <<'PY'
import time

from your_entry_module import create_app, create_main_window

t0 = time.perf_counter()
app = create_app()
t1 = time.perf_counter()
window = create_main_window()
t2 = time.perf_counter()

print(f'create_app: {t1 - t0:.3f}s')
print(f'create_main_window: {t2 - t1:.3f}s')
print(f'total: {t2 - t0:.3f}s')
PY
```

如果项目没有统一的工厂函数，可在入口文件中临时插入同类计时点，原则是保持测量边界清晰。

### 3.3 单独测试可疑模块的冷导入

当 `importtime` 或分段计时已经表明导入链过重时，建议对可疑模块逐个验证：

```bash
python3 - <<'PY'
import time

mods = [
    'your_heavy_module_a',
    'your_heavy_module_b',
    'your_heavy_module_c',
]

for mod in mods:
    start = time.perf_counter()
    __import__(mod)
    print(f'{mod}: {time.perf_counter() - start:.3f}s')
PY
```

此方法适合回答以下问题：

- 是否只有少数模块特别重。
- 哪些模块值得优先做懒加载。
- 重量级依赖是否被提前拉入启动路径。

### 3.4 区分“模块导入慢”与“对象构造慢”

启动优化中一个常见误判是：看到某个类相关代码很多，就默认问题出在构造函数。实际情况可能只是模块顶层导入过重。

可使用以下方式拆分：

```bash
python3 - <<'PY'
import time
from your_module import YourClass

start = time.perf_counter()
obj = YourClass()
print(f'YourClass init: {time.perf_counter() - start:.4f}s')
PY
```

判断原则：

- 如果模块导入很慢，而对象构造很快，应优先治理顶层导入。
- 如果模块导入较快，而对象构造很慢，应检查构造函数中的 I/O、扫描、模型加载或页面装配逻辑。

### 3.5 对 GUI 项目补充离屏 UI 测试

对于 PySide6 或 PyQt 项目，离屏测试有助于排除图形环境差异，并单独测量 UI 构造成本。

```bash
QT_QPA_PLATFORM=offscreen python3 - <<'PY'
import time

from PySide6.QtWidgets import QApplication
from your_entry_module import build_main_window

t0 = time.perf_counter()
app = QApplication([])
t1 = time.perf_counter()
window = build_main_window()
t2 = time.perf_counter()

print(f'QApplication: {t1 - t0:.3f}s')
print(f'build_main_window: {t2 - t1:.3f}s')
print(f'total: {t2 - t0:.3f}s')
PY
```

适用场景：

- 怀疑 `QApplication` 创建本身较慢。
- 怀疑主窗口或页面装配成本过高。
- 需要在无图形界面的自动化环境中复现问题。

说明：

- 示例中的 `build_main_window` 仅为占位名称，需要替换为实际构造逻辑。
- 如果窗口构造依赖配置对象、任务描述或服务容器，也应将这些准备步骤纳入计时。

### 3.6 对比源码运行与打包运行

源码分析和打包分析不应混为一谈。两者的启动开销来源并不相同。

建议至少对比以下维度：

- 源码运行与打包运行的总耗时差异。
- 首次启动与再次启动的差异。
- `--onefile` 与 `--onedir` 的差异。
- 是否存在首次解包、动态库加载、签名校验或安全扫描成本。

如果源码运行较快，而打包产物明显偏慢，优先检查打包方式和运行环境；不要仅依据源码分析结果修改业务代码。

## 4. 结果解读与常见优化方向

完成测量后，建议按“结论对应动作”的方式整理结果。

### 4.1 导入链明显偏重

常见现象：

- `import <入口模块>` 已经很慢。
- `importtime` 前几名集中在少数业务模块或第三方依赖。

常见处理方式：

- 对低频功能实施懒加载。
- 将重量级第三方依赖移出模块顶层。
- 减少启动阶段统一注册的功能数量。

### 4.2 初始化阶段明显偏重

常见现象：

- 模块导入速度正常，但应用启动后仍有明显停顿。
- 对象构造或初始化函数中存在磁盘扫描、网络访问、模型检测或缓存重建。

常见处理方式：

- 将初始化逻辑拆分为“必要初始化”和“延后初始化”。
- 避免在主窗口构造阶段执行大量 I/O。
- 对可缓存结果建立稳定缓存，而不是每次启动重新生成。

### 4.3 UI 构造明显偏重

常见现象：

- `QApplication` 或主窗口装配耗时明显。
- 页面、卡片、表格或复杂控件在启动阶段全部同步创建。

常见处理方式：

- 将非首屏页面改为按需构造。
- 推迟大数据量视图的初始化。
- 将首屏必须内容与次要内容分批加载。

### 4.4 打包与环境成本明显偏重

常见现象：

- 源码运行较快，打包后明显变慢。
- 首次启动明显慢于再次启动。
- 字体缓存、模型缓存或临时目录权限异常。

常见处理方式：

- 检查缓存目录是否可写。
- 调整打包方式，并验证 `--onefile` 是否引入额外启动成本。
- 记录并清理启动阶段产生的环境警告和异常回退逻辑。

## 5. 注意事项

为确保分析结果可信，建议同时注意以下事项：

- `importtime` 输出位于标准错误，未重定向时日志可能不完整。
- `tail` 适合预览，不适合替代完整排序和正式结论。
- 冷启动与热启动差异通常很大，应分别记录。
- 顶层导入可能触发副作用，例如创建缓存、扫描模型、写入临时文件或打印警告。
- GUI 项目的启动结论不应仅依赖主观体感，最好补充离屏测试或分段计时。
- 源码分析结果不能直接替代打包分析结果，两者必须分别验证。

## 6. 以 RD-tools 为例

本节用于说明上述方法如何落到当前项目。以下内容属于项目示例，不应直接视为其他项目的通用结论。

### 6.1 分析对象

本项目启动路径中，重点观察了以下部分：

- 入口模块 `main.py`。
- 主窗口相关模块 `main_window.py`。
- 若干任务模块，例如 `modules.ECG_handler`、`modules.gen_subtitles`、`modules.twist_shape`、`modules.dicom_to_imgs`。
- 打包方式与 Matplotlib 缓存环境。

### 6.2 实际观察

按本文方法拆分后，得到的主要结论如下：

- `import main` 明显偏慢，量级约为十几秒。
- `import main_window` 明显更快，不到一秒。
- 因此，当前瓶颈主要位于入口模块导入链，而不是主窗口外壳本身。

进一步拆分后，主要耗时集中在少数任务模块：

- `modules.ECG_handler`
  - 顶层导入了 `matplotlib`、`pandas`、`scipy`。
  - 单独冷导入为十秒级。
  - 其中 `matplotlib.font_manager` 是明显重项。

- `modules.gen_subtitles`
  - 顶层会拉起 `faster_whisper`，并进一步带出 `ctranslate2`、`torch`、`transformers`。
  - 单独冷导入通常为一秒到数秒级，受环境影响较大。

- `modules.twist_shape`
  - 顶层导入 `cv2`。
  - 存在明显成本，但不是当前最大瓶颈。

- `modules.dicom_to_imgs`
  - 顶层导入 `pydicom`、`PIL`、`numpy`。
  - 存在成本，但低于前述主要重项。

另外还观察到一个环境因素：

- Matplotlib 缓存目录不可写。
- 启动阶段会回退到临时目录创建缓存，甚至重建字体缓存。
- 该问题会显著放大首次启动和冷启动时间。

### 6.3 对本项目的建议优先级

基于当前分析结果，RD-tools 的优化优先级建议如下：

1. 先将任务模块改为按需导入，避免在应用启动时统一导入全部任务实现。
2. 再将模块内部的重量级依赖继续下沉，避免在模块顶层直接导入 `matplotlib`、`faster_whisper`、`cv2` 等依赖。
3. 修复 Matplotlib 缓存目录问题，避免重复构建字体缓存。
4. 如果发布版仍明显偏慢，再单独评估 `PyInstaller --onefile` 带来的启动成本。

对于当前项目，第一项通常最可能带来最明显的启动改善。

## 7. 建议的后续执行方式

如果后续再次出现启动耗时问题，建议固定执行以下流程：

1. 明确本次分析的测量口径。
2. 记录冷启动、热启动、源码运行和打包运行的基线。
3. 用分段计时判断慢点位于导入、初始化还是 UI 构造。
4. 用 `importtime` 和单模块冷导入锁定重量级模块。
5. 区分模块导入成本与对象构造成本。
6. 记录环境异常、缓存问题和打包差异。
7. 最后再决定优化方案与优先级。

按上述顺序执行，可以显著降低误判概率，并提高后续优化工作的收益。
