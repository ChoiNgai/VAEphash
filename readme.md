# 文档



## 开发文档

使用了开源的变分自编码框架

并在此基础上增加感知哈希算法的图像相似度比对



## 文件树目录

```
├─backup
├─cfgs
│  └─__pycache__
├─datasets
│  ├─images				%视频解码文件夹
│  └─images1			%视频解码文件夹（需要解码完一个视频之后将文件夹修改为此名称）
├─material
│  └─effects
└─modules
    ├─models			%自编码器
    │  └─__pycache__
    └─utils
        └─__pycache__
```



## 环境

系统：window10/linux/macos



> 编译环境: 
>
> python==3.7.4
>
> opencv-python
>
> argparse
>
> pytorch==0.4.1
>
> torchvision==0.2.2
>
> numpy==1.1.6
>
> matplotlib==3.1.1
>
> numpy==1.1.6



## 使用教程

- 跳转到文件根目录

  使用anaconda prompt并跳转到根目录

  例如文件在D:dev ，则输入 cd D:dev

  （若是windows系统，输入cd  D:dev之后还需输入D:）

- 编码

  anaconda需要安装程序所需模块，再输入 

  ```
  python train.py --videopath xxx.mp4
  ```

  其中xxx.mp4视频文件路径
  
  编码完成一个视频之后将datasets文件夹内的`images`文件夹更名为`images1`
  
  按照上面的命令继续编码第二个视频
  
- 比对

  在anaconda prompt输入

  ```
  python sim_anal.py
  ```

  

## 测试

使用8300H的CPU40%的资源，解码一个一小时的视频需要十分钟，比对47000张图片大概需要2分钟

将相同的视频分析，结果相似度全为1

分析结果图片保存在datasets文件夹