# Predict Future Plants

[![LICENSE](https://img.shields.io/badge/License-Apache--2.0-blue.svg?style=flat-square)](LICENSE)
[![Tensorflow](https://img.shields.io/badge/Tensorflow-1.13.1-yellow.svg?style=flat-square)](https://github.com/tensorflow)
[![Python3](https://img.shields.io/badge/Python-3.7.3-green.svg?style=flat-square)](https://github.com/topics/python)
[![Size](https://img.shields.io/badge/Size-31.4MB-%23ff4D5B.svg?style=flat-square)](https://github.com/DolorHunter/PredictFuturePlants/archive/master.zip)

**区域植被指数分析与物候预测** - 2019合肥工业大学建模校赛B题

- [在线论文](essay/区域植被指数分析与物候预测.pdf)
- [论文下载](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/DolorHunter/PredictFuturePlants/blob/master/essay/区域植被指数分析与物候预测.pdf)

**设计:**

CNN卷积神经网络建模. 喂入模型时间坐标, 输出图片. 时间坐标作为模型输入数据, 图像数据作为模型输出数据.

**使用:**

建立模型:

训练: 新建data目录将图片数据导入/data/Z1等. 运行backward.py训练模型.

测试: 运行test.py测试模型.

使用模型:

使用已有模型, 运行app.py输入时间坐标生成结果.

*[注意]: 源数据为[1200\*1200]tiff格式图片, 因算力问题模型运算结果设计为[64\*64], 损失像素点由插值算法补足, 故会带来较大精度上的问题.*

**工作明细:**

`感谢各位同僚的辛苦付出!!`

da\na|          [Wang Zixiang](https://github.com/DolorHunter)          |          [HfuterQ](https://github.com/hfuter7712)          |          [net0000](https://github.com/net0000)          
----|-----------------------|-----------------------|-----------------------
0615|   图片输入接口完成   |             -             |             -             
0616|     模型建立及调参    |             -             |             -             
0617|修正IO逻辑, 搭建测试框架|             -             |             -             
0618|修正前向逻辑 替换交叉熵算法|forward修正一处语法错误|             -             
0619|          -          |backward进行debug|             -          
0620|          -          |app修正两条语句的bug|             -       
0621|模型调整 写论文p1-4|app解决部分语句的bug|             -          
0622|写论文p4-5|修改app文件中的变量类型|已被强制离队 
0623|改论文p5 debug(f/b/a/t/f)|协助debug 处理图片 完成论文p6|             -             
