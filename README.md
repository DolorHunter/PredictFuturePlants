﻿# model_build_B

**设计:**

喂入时间输出图片. 手动对于图片进行解释分析. 撰写论文.

**使用:**

新建data目录将图片数据导入/data/Z1等.

运行backward.py训练模型.

运行test.py测试模型.

运行app.py生成结果.

**训练模型:**

图片的序号最为时间戳喂入模型(作为模型输入数据)
读图->转数组->变形喂入模型(作为模型输出数据)

**预测:**

模型预测->数组->变形->图->手动分析

**工作明细:**

da\na|          wzx          |          yyn          |          wyt          
----|-----------------------|-----------------------|-----------------------
0615|   图片输入接口完成   |             -             |             -             
0616|     模型建立及调参    |             -             |             -             
0617|修正IO逻辑, 搭建测试框架|             -             |             -             
0618|修正前向逻辑 替换交叉熵算法|对forward算法进行debug，修正一处语法错误|             -             
0619|          -          |对backward算法进行debug|             -          
0620|          -          |对app文件算法进行debug，修正两条语句的bug|             -       
0621|模型调整 写论文p1-4|对app文件运行出现的错误进行集体查询，得到若干解决方法，解决部分语句的bug|             -          
0622|写论文p4-5|修改app文件中的变量类型，查询变量类型转换相关函数，并进行变量转换，restore_model成功debug，运行数据|已被劝退本队 
0623|改论文p5 debug(f/b/a/t/f)|协助完成debug，并测试数据，处理图片，记录数据与模型，协助完成论文p6的撰写与校正|             -             