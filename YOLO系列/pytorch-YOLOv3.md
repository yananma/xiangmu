
YOLO 是 one-stage，速度快，精度不高。rcnn 系列是 two-stage  

IoU 和 map 概念解释，map 就是 precision recall 图的方形下所围的面积，是综合了 precision 和 recall。  

#### YOLOv1  

YOLOv1 是 one-stage，把检测问题转化为回归问题，直接一个 CNN 搞定。  

具体就是把图像分割成 SxS 个 grid，比如 7x7 的格子，然后物体本来就是打了标签的，物体的中心落在了哪个格子里，这个格子就负责这个物体的识别。一个格子先生成 2 个 anchor，再调整长宽。bbox 就是回归，4 个值 xywh，confidence 预测的是这个是不是物体。置信度低的就过滤掉了。  

v1 用的还是 GoogLeNet，输入大小是固定的，都是 448x448x3 的。最后成了 7x7x30，就是每一个格子有 30 个值，30 就是每一个 grid 有两个 anchor，每个 anchor 有 4 个回归值和 1 个 confidence，再加上 20 个类别，就是 30.

4 个损失函数，第一个是位置误差，位置误差中，有的物体大一些，有的物体小一些，大的物体损失值大一些，所以 wh 有一个根号，就是让大的影响小一些，让小的影响大一些。这是一个很粗糙的方式。后面两个是 confidence 损失，在 obj confidence 中，obj 标签是 1，noobj 标签是 0. 在 noobj confidence 中，noobj 标签是 1，obj 标签是 0. noobj 还有一个 λ 权重参数调整重要程度。最后是 20 分类的分类误差。  

不单是 v1 是这几个损失函数，后面的版本大体上也都是这样的。  

v1 的问题：只有 2 个 anchor，只能预测大物体，小物体检测不到；每个 grid 只负责一个物体的预测，物体重叠的时候效果就不行。   


#### YOLOv2  

加了 Batch Normalization，舍弃了 dropout。  

使用了更高的分辨率，v1 是训练时用的 224x224，测试时用的是 448x448，v2 是后面加了 10 次 448x448 的微调.  

网络结构用的是 darknet，没有 FC 层，只有 5 次 maxpooling。1x1 卷积省了很多参数，1x1 减少了 channel 数。5 次 maxpooling 使得最后和原来相差 32 倍。最后结果是 13x13 的，v1 版本是 7x7，所以 v2 就是网格更多了，网更细了，可以检测到小物体。  

提取先验框用了 kmeans，在 label 里聚类，每个 grid 有 5 个 anchor，anchor 多了以后提高了 recall 值。kmeans 的距离用的是 1 - IoU，因为框的大小不一样，用欧氏距离明显不好，IoU 大的距离小，IoU 小的距离大。

v2 使用的是相对坐标。YOLO 因为是一个 grid 只负责这个物体，所以用绝对坐标，可能会造成不收敛，会漂出 grid，所以用的是相对 grid cell 的坐标。

卷积核小一些好，小的卷积核效果一样，参数更少。可以用更多的 batch norm。  

Fine-Grained Features 不如后面的 FPN 好，意思是一样的。就是后面的感受野大，不容易探测小物体，就把后面和前面的融合在一起，办法就是前面的特征图拆分，再和后面的拼接。把前面 26x26 的特征图差分成 4 个 13x13 的特征图，再加上最后的 13x13 的特征图，所以最后就是 4x512+1024=3072.  

### YOLOv3  

工作以后继续写 YOLOv3  





PyTorch YOLOv3 没有用虚拟环境，如果基础配置有了较大的更改，就重新配置虚拟环境  

python 3.6  
pytorch 1.6.0  

#### train.py  

epochs 改成 6  
batch_size 改成 1  

