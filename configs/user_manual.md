# 0 前言

计算机擅长干重复的事情，而这恰恰是让我烦躁的，所以我花了半个月的业余时间做了这个软件的框架部分，能完成一些简单且枯燥的工作，虽然本来这些工作之需要几天，但是如果这能让更多人用到，也能让我欣慰一下，虽然我是挺乐在其中的。

## 0.0 该软件项目的由来

该软件是python编写的一款开源免费的用于科技领域工作者的小工具，当然可以推广到其他诸如金融等领域，但由于我暂不涉猎，无法了解其中的工作流程。

此软件要追溯到我的研究生时期，需要处理大量软件模拟的数据，于是我写了一些脚本自动生成模拟前的数据&自动处理模拟后的数据，并做成表格，但是实验室同学们不太懂计算机的无法使用该工具，于是我萌生了用UI封装这些脚本的想法。当时时间精力和知识储备都有所不足，做的东西完全得不到我自己的认可。但最近由于生物相关领域的同学问我一些批量化处理的问题，我决定做一个让我自己能打60分的适用于科研领域批量数据处理的软件框架。目前该软件还远不及60分，但是我看到了希望。

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

# 2. 应用

## 2.1 File Opt

如果有开发者查看源代码会发现，文件批量操作的逻辑都是类似的。下文以dicom处理和其他几个操作举例说明如何应用。

### 2.1.1 dicom处理

此模块功能：医学影像的dicom文件转换为图片或者视频，保存在与原dicomdir层级的文件夹中。

#### 2.1.1.1 UI处理

1. 首先选择工作目录，该目录为DICOMDIR文件的父目录的父目录。
2. 然后选择序号（DICOMDIR文件的父目录），假设有多个DICOMDIR文件的父目录（多次试验），可以空格隔开多选。
3. 点击提取选中序号
4. 点击《文件操作》的左侧sidebar列表中的DICOM处理
5. 点击右下方的同样名为“DICOM处理"按钮。
6. 观察“DICOM处理"按钮下方的日志显示，如果出现“SUCCESS! log file saved."，表示完成。

#### 2.1.1.2 CTL处理

在terminal中直接运行modules文件夹中的dicom_to_imgs.py文件，其与UI是相同的处理逻辑。

进一步来讲，如果想定制操作逻辑，比如你的DICOM文件结构和我假设的不同，可以重写dicom_to_imgs.py中最后的main函数，调用DicomToImage类、file_basic.py中基类的函数，实现针对你特殊文件结构中dicom文件的批量处理。
