# TODO

未来增加或修改的功能，如今设计计划不考虑此内容。

## 确认弹窗个性化预览

确认弹窗增加：个性化预览 比如 files_renamer 的前 10 条预览（但是通用组件提供基本的支持，限制传递的个性化预览信息内容个数，比如字符串个数小于50之类的）

## 安全性问题

1. 如果 loader 只是把这些对象原样传给新实例，后续某个任务一旦原地修改参数，就会把状态泄漏到下次运行。

2. 确认弹窗展示的是启动前的设置快照，但真正实例化参数是在工作线程里重新从 AppSettings 读取的，这意味着用户在确认后、线程开始前改设置，当前这次运行仍可能吃到新值，和弹窗内容不一致。应该在 UI 线程确认前后就把参数快照固定下来，并把该快照传给 BatchFilesBinding，不要在线程里再读 AppSettings？

3. 并发边界没有统一所有权，线程模型是层层叠加出来的。基类先在目录级并发，再在文件级并发，files_basic.py 开线程池；而具体任务又继续各自再开线程池，例如 dicom_to_imgs.py 、bili_videos.py。这会让 max_threads 的真实含义失真，GUI 响应、日志顺序、共享状态和第三方库线程安全都很难推理，ECG_handler.py 甚至已经只能靠“强制关并发”来规避问题。

4.  modules/app_settings.py 字符串校验过宽。实际复现里把 subtitle-generation.model_path 写成 false 后，effective value 会变成 'False'，而且 get_config_health() 为空。这和计划里“类型不匹配回退 schema default 并记录 issue”不一致。

