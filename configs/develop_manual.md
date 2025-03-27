# 0 前言

计算机擅长干重复的事情，而这恰恰是让我烦躁的，所以我花了半个月的业余时间做了这个软件的框架部分，能完成一些简单且枯燥的工作，虽然本来这些工作之需要几天，但是如果这能让更多人用到，也能让我欣慰一下，虽然我是挺乐在其中的。

## 0.0 该软件项目的由来

该软件是python编写的一款开源免费的用于科技领域工作者的小工具，当然可以推广到其他诸如金融等领域，但由于我暂不涉猎，无法了解其中的工作流程。

此软件要追溯到我的研究生时期，需要处理大量软件模拟的数据，于是我写了一些脚本自动生成模拟前的数据&自动处理模拟后的数据，并做成表格，但是实验室同学们不太懂计算机的无法使用该工具，于是我萌生了用UI封装这些脚本的想法。当时时间精力和知识储备都有所不足，做的东西完全得不到我自己的认可。但最近由于生物相关领域的同学问我一些批量化处理的问题，我决定做一个让我自己能打60分的适用于科研领域批量数据处理的软件框架。目前该软件还远不及60分，但是我看到了希望。

## 0.1 我的愿景

我很喜欢开源社区的氛围，尤其是从历史上来看，Unix到Linux，从git到github，我觉得没有计算机能发展之快就是不同于传统领域的开源精神。本项目的所有代码也都是基于开源的python和各种第三方库，所以我希望它能成为一款跨多领域的模块化的批量数据处理软件。

## 0.2 软件设计理念

我为此软件的模块化做了一些力所能及的努力，希望能帮助他在开源社区成长。包括但不限于：

- 每个功能的逻辑代码可以完全独立于UI代码作为脚本使用。
- 每个页面都可以单独运行显示，用于界面的微调。
- 尽可能简化功能界面和逻辑界面的绑定，且集成在main.py
- 实现了添加页面的“原子操作”。
- 实现了添加常用逻辑代码绑定的“原子操作”。

# 1 设计思路

## 1.0 架构

```python

this-project/
│
├── libs/
│
├── configs/
│   ├── develop_manual.md
│   ├── user_manual.md
│   └── settings.json
│
├── modules/
│   ├── __init__.py
│   ├── app_settings.py
│   ├── files_basic.py
│   ├── merge_colors.py
│   ├── split_colors.py
│   └── twist_shape.py
│   ├── dicom_to_imgs.py
│   └── serial_com.py
│   └── serial_plot.py
│   └── bili_videos.py
│   └── ECG_handler.py
│   └── gen_subtitles.py
│   └── sum_subtitles.py
│   └── AI_chat.py
│
├── widgets/
│   ├── __init__.py
│   ├── dock_widget.py
│   ├── setting_page.py
│   ├── file_page.py
│   ├── help_page.py
│   ├── plotting_page.py
│   └── images_page.py
│
├── main_window.py
├── main.py
├── requirements.txt
└── install.py
```

其中lib是可能依赖的动态库；configs内部为配置文件（显而易见）；modules内为逻辑代码部分；widgets内为UI页面的代码；main_window.py是主窗口的显示和初始化widgets内的页面类；main.py初始化main_window，绑定modules内的逻辑类。

## 1.1 从哪里讲起

一个程序都是从main开始，但是这是执行的开始，而不是设计的开始。我尝试从我设计此软件的思路讲，不知道效果会不会好一些。

## 1.2 批量处理的脚本

以下这些文件是批量处理的功能性脚本，每一个文件是一个独立的功能。
│   ├── merge_colors.py
│   ├── split_colors.py
│   └── twist_shape.py
│   └── bili_videos.py
│   └── dicom_to_imgs.py
│   └── ECG_handler.py
│   └── gen_subtitles.py
│   └── sum_subtitles.py

但由于其为批量处理，譬如扫描文件类型，创建多线程任务，捕捉错误或者正确消息传递给UI界面等功能是通用的，所以我抽象除了files_basic.py，其中的类FilesBasic作为这些功能性文件的基类：

1. 重写single_file_handler函数就可以集成FilesBasic的批量处理流程中。
2. 在main函数中仿照其他几个类实例化并绑定几个信号连接就可以将该功能集成到UI中。
3. 在AppSettings和setting.json中仿照其他几个类的参数添加你的类的初始化参数，就可以实现在UI界面中修改参数的功能。
