# RD-tools 启动耗时分析指南

本文档用于说明 RD-tools 当前这类桌面 Python 应用，应该如何分析“启动慢”的原因，描述分析方法、判断思路和后续优化方向。

## 1. 最先执行的命令

启动耗时分析时，第一步建议先跑下面这条命令：

```bash
python3 -X importtime -c 'import main' 2> /tmp/rdtools_importtime.txt && tail -n 120 /tmp/rdtools_importtime.txt
```

这条命令的含义是：

- `python3 -X importtime`
  - 打开 Python 自带的 import 耗时统计。
- `-c 'import main'`
  - 直接导入入口模块，模拟程序启动时的 import 链路。
- `2> /tmp/rdtools_importtime.txt`
  - `importtime` 输出写在标准错误里，所以要重定向到文件。
- `tail -n 120`
  - 先看最后 120 行，快速观察末端的大头模块。

这个命令适合“先抓一把大概情况”，但它还不够通用，也不够适合后续反复使用。

## 2. 更通用、更友好的版本

### 2.1 通用函数：指定入口模块并保存结果

推荐在终端里临时定义下面这个函数：

```bash
profile_importtime() {
  local module="${1:-main}"
  local out="${2:-/tmp/${module}_importtime_$(date +%Y%m%d_%H%M%S).txt}"

  python3 -X importtime -c "import ${module}" 2> "${out}" || return $?

  echo "importtime 日志: ${out}"
  echo
  tail -n 120 "${out}"
}
```

用法示例：

```bash
profile_importtime
profile_importtime main
profile_importtime main_window
profile_importtime modules.ECG_handler
```

这个版本比单条命令更友好，原因是：

- 可以直接传入任意模块名
- 输出文件带时间戳，不会反复覆盖
- 默认参数对当前项目足够顺手
- 适合后续做多轮对比

### 2.2 汇总函数：按累计耗时排序

只看 `tail -n 120` 不够，因为它只能看到日志尾部，不能稳定找出“累计耗时最大”的模块。建议再配一个汇总函数：

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

用法示例：

```bash
profile_importtime main /tmp/rdtools_importtime.txt
summarize_importtime /tmp/rdtools_importtime.txt
```

建议把这两个函数一起用：

1. `profile_importtime` 先生成原始日志。
2. `summarize_importtime` 再按累计耗时排序看前 30 名。

## 3. 推荐的分析顺序

分析启动慢时，不要一上来就假设“模块太多”或者“UI 太复杂”。建议按下面顺序拆：

### 3.1 先区分是 import 慢，还是窗口构造慢

先测入口 import：

```bash
python3 - <<'PY'
import time
start = time.perf_counter()
import main
print(f'import main: {time.perf_counter() - start:.3f}s')
PY
```

再测主窗口模块 import：

```bash
python3 - <<'PY'
import time
start = time.perf_counter()
import main_window
print(f'import main_window: {time.perf_counter() - start:.3f}s')
PY
```

如果 `import main` 很慢，但 `import main_window` 明显快，通常说明问题主要在业务模块导入链，而不是窗口壳本身。

### 3.2 再拆单个重量级模块

对怀疑重的模块逐个做冷导入：

```bash
python3 - <<'PY'
import time

mods = [
    'modules.ECG_handler',
    'modules.gen_subtitles',
    'modules.twist_shape',
    'modules.dicom_to_imgs',
]

for mod in mods:
    start = time.perf_counter()
    __import__(mod)
    print(f'{mod}: {time.perf_counter() - start:.3f}s')
PY
```

这样可以快速确认：

- 是不是只有少数模块特别重
- 哪些功能最值得优先做懒加载

### 3.3 再区分“模块导入慢”还是“对象构造慢”

有时类的 `__init__()` 很重，有时只是模块顶层 import 很重。需要拆开看：

```bash
python3 - <<'PY'
import time
from modules.ECG_handler import ECGHandler

start = time.perf_counter()
ECGHandler()
print(f'ECGHandler init: {time.perf_counter() - start:.4f}s')
PY
```

如果对象构造几乎不耗时，而模块导入很慢，说明应该优先治理顶层 import，而不是先改构造函数。

### 3.4 如果怀疑 UI 构造慢，再测窗口装配

GUI 项目可以用离屏模式做无界面测试：

```bash
QT_QPA_PLATFORM=offscreen python3 - <<'PY'
import time
import main
from PySide6.QtWidgets import QApplication

start = time.perf_counter()
app = QApplication([])
settings = main.AppSettings()
descriptors = main.build_task_descriptors()
window = main.MainWindow(settings, descriptors)
print(f'build ui: {time.perf_counter() - start:.3f}s')
PY
```

这样可以判断慢点是在：

- Python import 链
- Qt 应用对象创建
- 主窗口及页面装配

### 3.5 最后再看打包方式有没有叠加成本

如果源码启动和打包启动体感差异很大，还要检查打包模式，尤其是：

- `PyInstaller --onefile`
- 首次启动自解包
- 动态库加载
- 安全扫描

源码分析只能覆盖 Python 层；打包产物的启动开销要单独看。

## 4. 本次 RD-tools 的分析思路

这次分析 RD-tools 启动慢时，采用的是下面这条链路：

1. 先看入口文件 `main.py`
   - 判断是否在顶层直接导入了所有任务模块。
   - 如果顶层已经把所有任务类都 `import` 进来，那么懒加载就很可能有意义。

2. 再看 `main_window.py` 和页面构造
   - 判断 UI 是否在启动时主动扫描工作目录、初始化任务数据、或做重 I/O。
   - 如果 UI 只是创建列表和卡片，一般不是主因。

3. 使用 `python3 -X importtime -c 'import main'`
   - 把启动导入链完整记录下来。
   - 再用排序脚本找出累计耗时最大的模块。

4. 对重模块做单独冷导入测试
   - `modules.ECG_handler`
   - `modules.gen_subtitles`
   - `modules.twist_shape`
   - `modules.dicom_to_imgs`

5. 再测类实例化耗时
   - 验证慢点是在模块顶层，还是在对象构造阶段。

6. 最后补充环境和打包因素判断
   - Matplotlib 字体缓存
   - 模型依赖是否在 import 时被拉起
   - `PyInstaller --onefile` 是否额外放大启动时间

## 5. 本次 RD-tools 的主要发现

这次分析里，结论比较明确：

- `import main` 明显慢，量级约十几秒
- `import main_window` 只有不到一秒
- 所以慢点主要不在窗口壳，而在入口模块的导入链

继续拆之后，主要瓶颈集中在少数任务模块：

1. `modules.ECG_handler`
   - 顶层导入了 `matplotlib`、`pandas`、`scipy`
   - 单独冷导入大约十秒级
   - 其中 `matplotlib.font_manager` 非常重

2. `modules.gen_subtitles`
   - 顶层尝试导入 `faster_whisper`
   - 进一步带出 `ctranslate2`、`torch`、`transformers`
   - 单独冷导入约一秒到数秒级，视环境而定

3. `modules.twist_shape`
   - 顶层导入 `cv2`
   - 有成本，但比前两项小

4. `modules.dicom_to_imgs`
   - 顶层导入 `pydicom`、`PIL`、`numpy`
   - 有成本，但不是当前最大头

另外还发现一个环境问题：

- Matplotlib 当前缓存目录不可写
- 导致启动阶段会在临时目录创建缓存，甚至重建字体缓存
- 这会显著放大首次启动或冷启动时间

因此，本项目当前启动慢的主要原因不是“`modules` 文件多”，而是“少数重量级功能在程序启动时被统一提前导入”。

## 6. 常见解决方案

### 6.1 优先做任务级懒加载

这是当前项目最值得优先考虑的方案。

典型做法是：

- 任务描述里不直接保存类对象
- 改为保存模块路径和类名
- 用户真正点击执行某个任务时，再 `importlib.import_module()` 加载对应实现

这样做的好处是：

- 不会因为一个很少使用的功能拖慢整个应用启动
- 可以把优化集中在重模块，不需要平均对待所有模块

### 6.2 把重量级依赖移出模块顶层

即使做了任务级懒加载，有些模块内部仍然可以继续优化。

例如：

- 把 `matplotlib.pyplot` 移到真正需要画图的方法里
- 把 `faster_whisper` 移到真正开始转录时再导入
- 把 `cv2` 移到真正进行图像变换时再导入

这一步的目标是：

- 把“模块可导入”和“功能真正执行”分开
- 让模块注册阶段更轻

### 6.3 修正缓存目录和环境问题

例如：

- 为 Matplotlib 提供可写缓存目录
- 检查 `MPLCONFIGDIR`
- 避免每次启动都重建字体缓存

这类问题不解决，代码已经优化后，仍然可能觉得冷启动偏慢。

### 6.4 重新评估打包方式

如果桌面版本主要给自己长期使用，且特别关注启动速度，可以评估：

- 是否一定要 `--onefile`
- 是否改用 `--onedir`

`--onefile` 的优点是分发方便，但启动一般比 `--onedir` 更慢。

### 6.5 必要时把超重任务独立进程化

如果某些任务依赖栈特别重，而且与主程序生命周期并不强绑定，可以考虑：

- 主程序保持轻量
- 真正运行任务时再启动子进程

这适合：

- 模型类任务
- 科学计算类任务
- 媒体处理类任务

但这一步复杂度更高，通常排在懒加载之后。

## 7. 注意事项

### 7.1 `importtime` 输出在标准错误

不要忘了用 `2>` 重定向，否则日志可能看起来不完整。

### 7.2 `tail -n 120` 只是快速预览

它适合“先扫一眼”，不适合正式判断。正式分析要看完整日志，并做排序汇总。

### 7.3 冷启动和热启动结果会不同

例如：

- 字体缓存
- Python 字节码缓存
- 动态库缓存
- 磁盘缓存

所以建议至少区分：

- 首次冷启动
- 再次启动

### 7.4 顶层 import 可能有副作用

有些模块在 import 阶段就会：

- 创建缓存
- 检查环境
- 扫描模型
- 打印警告
- 写临时文件

所以分析时要留意，别把这些副作用误认为业务执行本身。

### 7.5 GUI 项目最好补离屏测试

有些环境下直接构造 `QApplication` 会受图形环境影响，建议用：

```bash
QT_QPA_PLATFORM=offscreen
```

这样更适合自动化测量。

### 7.6 源码分析不等于打包分析

源码下 `import main` 快，不代表打包后也快；反过来也一样。两者都要单独验证。

## 8. 后续标准排查步骤

后续如果再遇到“启动慢”问题，建议固定按下面流程执行：

1. 运行 `profile_importtime main`
   - 先拿到完整 import 日志。

2. 运行 `summarize_importtime`
   - 先找累计耗时最大的模块，不靠肉眼猜。

3. 单独测试主窗口 import
   - 判断问题在业务导入链，还是 UI 模块。

4. 单独测试重模块冷导入
   - 锁定最值得优化的 2 到 3 个模块。

5. 测类实例化耗时
   - 判断应该优先改顶层 import，还是改构造函数。

6. 补充离屏 UI 测试
   - 排除 `QApplication` 和页面装配带来的影响。

7. 记录环境异常
   - 例如缓存目录不可写、动态库警告、模型路径扫描等。

8. 最后再决定方案优先级
   - 先做懒加载
   - 再做局部重依赖延迟导入
   - 再处理缓存和打包问题

## 9. 对当前项目的建议优先级

基于本次分析，当前项目建议优先级如下：

1. 先把任务模块改成按需导入
   - 重点是 `ECG_handler`、`gen_subtitles`、`twist_shape`

2. 再把模块内部的重依赖继续下沉
   - 重点是 `matplotlib`、`faster_whisper`、`cv2`

3. 修复 Matplotlib 缓存目录问题

4. 如果发布版仍慢，再评估 `PyInstaller --onefile`

这四步里，第一步最关键，也最可能带来最明显的启动改善。
