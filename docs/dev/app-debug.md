# RD_Tool APP 调试思路

## 一、MacOS 启动调试思路

### 背景

- 平台: macOS arm64
- Python: 3.12.11
- PyInstaller: 6.15.0
- 打包模式: `--onedir --windowed`
- 现象: `PyInstaller` 日志显示构建成功, 但 Finder 打开 `dist/RD_Tool.app` 后没有界面, 应用很快退出

### 这次排查的核心思路

不要停留在“构建成功/失败”层面, 重点判断应用在 macOS 上到底是:

1. 没有被 LaunchServices 拉起
2. 被拉起了但被 Gatekeeper 或签名拦住
3. 已经启动, 但在 Python 入口阶段因为资源/路径问题直接退出

这三类问题的排查路径完全不同。

### 实际排查步骤

#### 1. 先确认问题不是 `PKG`

`PyInstaller` 日志里的:

```text
Building PKG (CArchive) RD_Tool.pkg
```

这里的 `PKG` 是 `PyInstaller` 的内部归档, 不是给用户双击打开的 macOS 安装包。真正应该验证的是:

- `dist/RD_Tool.app`
- `dist/RD_Tool/RD_Tool`

#### 2. 先确认 `.app` 是否真的被 macOS 拉起

用:

```bash
open -W dist/RD_Tool.app
```

如果命令几乎立刻返回, 说明应用不是“后台活着但没窗口”, 而是启动后很快退出。

这次排查里, `open -W` 是很快返回的, 所以可以直接判定应用在启动后秒退。

#### 3. 用系统日志抓真实退出原因

最有价值的命令是:

```bash
log show --style compact --last 1m --predicate 'process == "RD_Tool" OR senderImagePath CONTAINS "RD_Tool" OR eventMessage CONTAINS "RD_Tool"'
```

这一步拿到了真正的报错:

```text
Failed to execute script 'main' due to unhandled exception:
[Errno 2] No such file or directory:
'/Users/zhengxu/Desktop/some_code/my-personal-git/RD-tools/dist/RD_Tool.app/Contents/Frameworks/ui/qss/dark/app.qss'
```

这个错误说明:

- 应用已经被 Finder / LaunchServices 正常拉起
- 也进入了 Python 主程序
- 不是签名校验本身导致的直接拦截
- 真正的主因是运行时资源文件路径错误

### 已经确认的结论

#### 结论 1: 主因不是 SQLAlchemy / ctypes 那些 warning

以下 warning 在这次问题里不是主因:

- `Hidden import "pysqlite2" not found!`
- `Hidden import "MySQLdb" not found!`
- `Hidden import "psycopg2" not found!`
- `Ignoring AppKit.framework/AppKit ...`
- `Library user32/dwmapi/msvcrt required via ctypes not found`

这些都发生在 `Analysis` 阶段, 但最终真正导致应用退出的是 `ui/qss/dark/app.qss` 缺失。

#### 结论 2: 主因是打包时没有把静态资源带进来, 同时代码按源码目录方式取资源

当前项目里至少有 4 处运行时会读取仓库内静态资源:

- `ui/theme.py`
- `widgets/file_page.py`
- `widgets/help_page.py`
- `modules/app_settings.py`

如果只修 `theme.py`, 后续仍然可能在这些位置再次触发 `FileNotFoundError`。

#### 结论 3: 构建脚本还存在一个会影响复现效率的问题

`install.py` 用的是非交互式 `subprocess.check_call(...)`, 但没有给 `PyInstaller` 传 `--noconfirm`。

结果是:

- 如果 `dist/` 已存在
- `PyInstaller` 需要覆盖旧产物
- 它会尝试询问是否删除目录
- 在当前脚本环境里没有交互输入
- 最终触发 `EOFError`

这个问题不会导致“打不开”, 但会让重建和验证过程非常低效。

### 这次已经做过的修复尝试

以下修复已经落到代码里:

1. 新增统一资源解析 helper:

   - `core/resource_paths.py`

2. 让以下模块不再直接假设资源一定在源码目录旁边:

   - `ui/theme.py`
   - `widgets/file_page.py`
   - `widgets/help_page.py`
   - `modules/app_settings.py`

3. 在 `install.py` 里给 `PyInstaller` 增加数据目录打包:

   - `ui/qss`
   - `configs`

4. 在 `install.py` 里增加 `--noconfirm`, 避免重建时卡在目录覆盖确认

### 当前还没完成的验证

由于完整重打包较慢, 这一轮在用户要求先记录思路后中断了, 所以以下验证还需要继续做:

1. 完整重跑一次打包
2. 确认新的 `.app` 内已经包含:

   - `ui/qss`
   - `configs`

3. 再执行:

```bash
open -W dist/RD_Tool.app
```

如果命令不再立刻返回, 说明“秒退”主因已经解决。

### 下一轮建议的验证顺序

#### A. 先准备隔离的 build venv

```bash
python install.py setup-build-env
```

#### B. 再重新打包

```bash
python install.py build
```

如果走交互菜单:

1. 先选 `2` 创建或更新 build venv
2. 再选 `3` 打包

#### C. 先看资源是否真的进包

```bash
find dist/RD_Tool.app -path '*ui/qss*' -o -path '*configs*' | sort
```

#### D. 再测启动行为

```bash
open -W dist/RD_Tool.app
```

#### E. 如果还退出, 再抓系统日志

```bash
/usr/bin/log show --style compact --last 1m --predicate 'process == "RD_Tool" OR senderImagePath CONTAINS "RD_Tool" OR eventMessage CONTAINS "RD_Tool"'
```

### 后续可以继续优化, 但不是这次主因

下面这些问题值得后续处理, 但不应和“打不开”混在一起:

- 当前 Python 环境过于臃肿, `PyInstaller` 拉进了很多与主程序无关的包
- `rapidfuzz.__pyinstaller:get_hook_dirs` 的 warning 还在
- `matplotlib` 缓存目录不可写会拖慢分析阶段
- 如果未来要分发给其他机器, 还需要处理 Developer ID 签名和 notarization

## 二、动态导入任务模块缺失

### 现象

- 源码运行 `python main.py` 正常
- 打包后的 macOS 和 Windows 版本都能启动主界面
- 但执行“批量重命名”任务时提示:

```text
任务执行失败: No module named 'modules.files_renamer'
```

这类问题和前面的“应用启动即退出”不是同一类故障。

- 前者是应用已经启动成功, 但在运行某个任务时动态导入失败
- 后者是应用在 Python 入口阶段就直接退出

### 根因

任务实现不是在主入口里静态 `import` 的, 而是通过任务注册表里的字符串模块路径动态加载。

当前链路是:

- `core/task_registry.py` 里把任务模块登记为字符串, 例如 `modules.files_renamer`
- `core/task_loader.py` 里用 `importlib.import_module(module_path)` 按字符串导入

源码运行时, Python 会直接从仓库目录找到这些模块, 所以没有问题。

但 `PyInstaller` 的静态分析默认看不到这类“运行时按字符串导入”的模块, 所以如果不显式声明 hidden import:

- 打包时这些任务模块可能不会被收进产物
- 最终表现为应用能打开, 但点击任务后才报 `No module named ...`

### 为什么只报 `modules.files_renamer`

`modules.files_renamer` 只是最先被触发的任务模块之一, 不是唯一风险点。

只要某个任务模块满足下面两个条件, 都可能出现同类问题:

1. 模块路径只存在于注册表字符串里
2. 主程序里没有静态导入它

所以这个问题本质上不是“重命名模块特殊”, 而是“整个任务动态导入机制需要对打包器显式声明”。

### 解决思路

不要手工一个个补 `--hidden-import=modules.xxx`。

更稳妥的做法是:

1. 从任务注册表统一读取所有 `module_path`
2. 在 `install.py` 生成 `PyInstaller` 命令时, 自动为每个任务模块追加 `--hidden-import=<module_path>`

这样做的好处是:

- 新增任务时不需要再同步维护打包脚本
- macOS / Windows / Linux 都复用同一套逻辑
- 不会只修 `files_renamer`，却把别的任务模块继续漏掉

### 这次实际修复

这次修复包含两部分:

1. 在 `install.py` 中新增从任务注册表收集模块路径的逻辑, 并统一追加 hidden imports
2. `build` 命令改为优先复用现有 build venv, 同时把 `PYINSTALLER_CONFIG_DIR` 和 `MPLCONFIGDIR` 指到仓库内可写目录, 避免再次被用户目录缓存权限影响构建

修复后重新打包时, 日志里应能看到类似内容:

```text
Analyzing hidden import 'modules.files_renamer'
Analyzing hidden import 'modules.bili_videos'
Analyzing hidden import 'modules.gen_subtitles'
...
```

### 修复后的验证方式

#### 1. 重新打包

```bash
python install.py build
```

#### 2. 看构建日志里是否真的分析了目标模块

关键字:

- `Analyzing hidden import 'modules.files_renamer'`

#### 3. 再看交叉引用文件是否包含该模块

可检查:

```bash
rg -n "modules\\.files_renamer" build/RD_Tool/xref-RD_Tool.html
```

如果能看到 `modules.files_renamer` 的条目, 说明它已经进入这次打包分析结果。

#### 4. 最后再做运行验证

- 启动应用
- 执行“批量重命名”任务
- 确认不再出现 `No module named 'modules.files_renamer'`

### 这一类问题的判断经验

如果遇到下面这种组合, 应优先怀疑 `PyInstaller` 漏收动态导入模块:

- 源码运行正常
- 打包产物主界面也能打开
- 只有点某个功能时才报 `No module named xxx`

这时不要优先去查业务逻辑异常, 应先确认:

- 该模块是不是通过字符串动态导入
- `PyInstaller` 命令里是否显式声明了 hidden import
- 打包分析产物里是否真的收进了该模块
