# 1 架构

软件由几大部分组成，分别对应Dock栏中的分组和按钮。

- setting_help部分：
  - help window：就是查看帮助文档，包括本文档和开发者文档。
  - setting window：用于修改改软件的大部分设置。
- image opt部分：（待开发）
  - Plotting window：用于通信过程中获取数据的实时波形显示。
  - Data Plot window：用于现有数据的图表制作。
  - image handler window：用于单张图片的常用操作。
- File opt部分：
  - File Window：用于批量操作多个文件夹
  - Atom File Opt Window：用于调试单个文件夹内的操作（调试成功后集成到File Window）

# 2. File Opt
如果有开发者查看源代码会发现，文件批量操作的逻辑都是类似的。下文以dicom处理和其他几个操作举例说明如何应用。

## 2.0 设置
设置文件保存到：用户文件夹/Develop/RD-tools-configs/settings.json
也可以在UI界面修改，注意有的参数可能会重启软件才会生效，多数参数可以即时生效。

## 2.1 dicom处理
此模块功能：医学影像的dicom文件转换为图片或者视频，保存在与原dicomdir层级的文件夹中。

### 2.1.1 UI处理
1. 首先选择工作目录，该目录为DICOMDIR文件的父目录的父目录。
2. 然后选择序号（DICOMDIR文件的父目录），假设有多个DICOMDIR文件的父目录（多次试验），可以空格隔开多选。
3. 点击提取选中序号
4. 点击《文件操作》的左侧sidebar列表中的DICOM处理
5. 点击右下方的同样名为“DICOM处理"按钮。
6. 观察“DICOM处理"按钮下方的日志显示，如果出现“SUCCESS! log file saved."，表示完成。

### 2.1.2 CTL处理
在terminal中直接运行modules文件夹中的dicom_to_imgs.py文件，其与UI是相同的处理逻辑。

进一步来讲，如果想定制操作逻辑，比如你的DICOM文件结构和我假设的不同，可以重写dicom_to_imgs.py中最后的main函数，调用DicomToImage类、file_basic.py中基类的函数，实现针对你特殊文件结构中dicom文件的批量处理。


## 2.2 ECG信号处理
此模块功能：将电极（单个导联）的数据采集到上位机保存成CSV文件。采样率（默认1000Hz）设定成自己数据的采样率，选择数据所在文件夹的上一级目录作为工作目录，选择文件夹序号，就可以把选择的文件夹内的所有CSV数据处理，生成以下几类图片：
1. 原始数据时域/频域图

2. 一级滤波后的时域/频域/对照图

3. 高级处理图(R波峰值检测、PQRST波形检测、心率计算)。


## 2.3 B站视频导出
此功能是将B站app缓存的文件批量生成，类似yt-dlp，不过可以处理缓存视频，且可以一次性全部导出为mp4。

参考：https://github.com/molihuan/BilibiliCacheVideoMergePython


## 2.4 字幕生成 & 字幕总结
参考：https://github.com/WEIFENG2333/VideoCaptioner

VideoCaptioner这个项目已经做的非常好了，我也用过一段时间，但是没有Linux和Mac版，且需要安装额外的whisper模型文件。还有翻译的功能我用不到，反而是需要大模型总结的需求，所以我就根据自己的需求写了一个简单版。配置上电脑中有的whisper模型文件地址就可以(比如设置成potplayer中的faster-whisper模型)。当然代价就是需要自行配置whisper-cpp（建议linux or mac）或者faster-whisper（建议windows）。

字幕总结还未完成...


## 2.5 图片RGB通道分解/合成
即 SplitColors 和 MergeColors。并不是只有荧光标记的图片可以操作，所有图片都可以，但一般图片没有这种需求。

MergeColors：批量搜索成对的R G B, 并把它们红绿通道合成为图片保存。当然可以指定只有R、G之类的。SplitColors则反之。


## 2.6 Com_Driver & SerialPlot
这是一个基于串口通信的通信基类和一个实时画图的类，本来没有找到比较好用的串口通信工具，打算自己写一个，但后来找到了Serial Studio就搁置了。之后有时间和契机的话，基于它和ECG信号处理模块做一个类似简易心电图机。


## 2.7 TwistImgs
这个是我个人的小需求，并不通用，是一个把图片处理成视觉上有一定角度的畸变的四边形，适用于搭建个人的海报场景等。