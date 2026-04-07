# RD_Tool APP 调试思路

## MacOS 启动调试思路

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
