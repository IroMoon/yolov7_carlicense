本代码是在yolov7代码的基础上修改的：官方链接 https://github.com/WongKinYiu/yolov7
车牌检测借鉴 https://github.com/we0091234/yolov7_plate
本代码仅用于交流学习。 如需引用请参考以上源代码链接。

# 1 食用方法【人工智能班级的学生】***
## 1. vscode用户可鼠标点击我，然后按住ctrl + shift + V 进入阅读模式
## 2. 先测试 5 检测  部分
## 3. 运行 4 迁移学习 部分，利用自己的数据进行训练

end



## YOLOv7 目标检测 与 车牌识别

Implementation of paper - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)


## Performance 

MS COCO

| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | batch 1 fps | batch 32 average time |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| [**YOLOv7**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) | 640 | **51.4%** | **69.7%** | **55.9%** | 161 *fps* | 2.8 *ms* |
| [**YOLOv7-X**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) | 640 | **53.1%** | **71.2%** | **57.8%** | 114 *fps* | 4.3 *ms* |
|  |  |  |  |  |  |  |
| [**YOLOv7-W6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt) | 1280 | **54.9%** | **72.6%** | **60.1%** | 84 *fps* | 7.6 *ms* |
| [**YOLOv7-E6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) | 1280 | **56.0%** | **73.5%** | **61.2%** | 56 *fps* | 12.3 *ms* |
| [**YOLOv7-D6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt) | 1280 | **56.6%** | **74.0%** | **61.8%** | 44 *fps* | 15.0 *ms* |
| [**YOLOv7-E6E**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt) | 1280 | **56.8%** | **74.4%** | **62.1%** | 36 *fps* | 18.7 *ms* |

# 2 环境安装 ***


<details><summary> <b>展开</b> </summary>

``` shell
# 0.解压代码

# 1.安装pytorch，默认电脑没有独立显卡（安装很快）【下一步的pytorch不用装】
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

# 1.安装pytorch,【电脑有独立显卡才装这个，否则不装】【必须使用anaconda安装】
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 2.打开anaconda或cmd，定位到代码所在位置
cd XX/XX/yolov7

# 3.安装其他库
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 环境配置完成
```
</details>

# 3 测试--运行目标检测，测试结果 【此操作先跳过】***
## 1. yolov7训练好的模型如下，可自行下载。【此步骤可忽略】
[`yolov7.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) [`yolov7x.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) [`yolov7-w6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt) [`yolov7-e6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) [`yolov7-d6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt) [`yolov7-e6e.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt)

## 2. anaconda界面输入下面命令运行。
   【说明：运行这个代码会测试目标检测的效果，data/coco.yaml文件中的val数据会用于测试。因此你需要将图片按照格式保存在对应的文件夹中】
   【运行结果保存在runs/test/文件夹中】
``` shell
python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val
```


# 4 迁移训练 ***
## 1. 如果没有yolov7_training.pt，点下面链接下载【已经下载好了】
[`yolov7_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt) [`yolov7x_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x_training.pt) [`yolov7-w6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6_training.pt) [`yolov7-e6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6_training.pt) [`yolov7-d6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6_training.pt) [`yolov7-e6e_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e_training.pt)

## 2. 数据准备  【运行已准备好的车牌数据，以下内容均无需改动】
### 2.1车牌数据。【已经准备好了, 再文件夹dataset_carlic中】
####  2.1.1
    dataset_carlic文件夹中,有images,labels两个文件，其中images中存放车牌图片， labels中存放车牌标签。
    还有train.txt, val.txt，generate_txt.py文件。
    train.txt， val.txt记录图片的路径。
    若想增加自己的数据，可以将图片放到images中，并放对应的标签到labels中，
    并将新增的图片路径添加到train.txt中。可以运行generate_txt.py自动添加。
####  2.1.2 修改 data/custom.yaml 文件
    里面的train val test 改成对应的train.txt 等的位置【默认不用改】


### 2.2训练你自己的数据
#### 2.2.1
    将你的训练数据放到文件夹【自己新建一个文件夹】中，里面的内容格式和dataset_carlic文件夹一样即可。
#### 2.2.2  修改 data/custom.yaml 文件
    里面的train val test 改成对应的train.txt 等的位置


## 3. 训练
``` shell
# finetune p5 models
python train.py --workers 8 --device 0 --batch-size 8 --data data/car_license.yaml --img 640 640 --cfg cfg/training/yolov7_carl.yaml --weights checkpoint/yolov7_training.pt --name yolov7_carl --hyp data/hyp.scratch.custom.yaml
```
说明： --batch-size 8 表示一次同时训练8张图片，可根据电脑的性能调整大小。

# 5 预测【最简单-可以先运行这个】***

视频检测:【需要将source后面的yourvideo.mp4改成你存放视频的位置】
``` shell
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source yourvideo.mp4
```

图片检测:【可直接运行，检测完成图片保存在runs/exp/img中】【80个类别的目标检测】
``` shell
python detect.py --weights checkpoint/yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
```

车牌识别：【进行车牌检测并识别出车牌的文字，结果保存在runs/exp/img中】【同时保存车牌结果在1124x.txt文件中】
``` shell
python detect_plate_car_lin.py --weights checkpoint/car_024.pt --car-txt 1124x.txt --rec_model checkpoint/ocr_0.77.pth --conf 0.4 --img-size 640 --source dataset_carlic/images/val/1.jpg
```

<div align="center">
    <a href="./">
        <img src="./figure/horses_prediction.jpg" width="59%"/>
    </a>
</div>


## detect.py 说明【可忽略】   
检测车牌的位置

pred = model()
pred 的形状为(1,n,6)  1 为batch-size， n为目标检测框形状，6：前4个为中心x,y与宽高w,h，5为置信度，6为分类

non_max_suppression() 删掉一些目标框，目标框特别多，很多重复


--save-license  保存车牌图片
--save-img 保存图片（标注好车牌）

README.md

