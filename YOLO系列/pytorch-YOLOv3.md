
YOLO 是 one-stage，速度快，精度不高。rcnn 系列是 two-stage  

IoU 和 map 概念解释，map 就是 precision recall 图的方形下所围的面积，是综合了 precision 和 recall。  

#### YOLOv1  

YOLOv1 是 one-stage，把检测问题转化为回归问题，直接一个 CNN 搞定。  

具体就是把图像分割成 SxS 个 grid，比如 7x7 的格子，然后物体本来就是打了标签的，物体的中心落在了哪个格子里，这个格子就负责这个物体的识别。一个格子先生成 2 个 anchor，再调整长宽。bbox 就是回归，4 个值 xywh，confidence 预测的是这个是不是物体。置信度低的就过滤掉了。  

v1 用的还是 GoogLeNet，输入大小是固定的，都是 448x448x3 的。经过卷积层和全连接以后，最后都 reshape 成了 7x7x30，就是每一个格子有 30 个值，30 就是每一个 grid 有两个 anchor，每个 anchor 有 4 个回归值和 1 个 confidence，再加上 20 个类别，就是 30.

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

YOLOv3 不像 v2 版本那样改进了这么多的细节，v3 主要改进的是网络结构，使得网络可以提取出更多的特征，从而提高了在小目标上的检测效果  

anchor 更丰富了，使用了 multi scale，有 3 种 scale，13x13、26x26、52x52，每种 scale 有 3 种 shape，这些 shape 也是用 kmeans 聚类得到的，一个坐标有 9 种 anchor  

还把整体信息向下融入了局部特征图中，出发之前看地图一样  

softmax 改进，一个物体可以打很多种标签，原来只能预测一种，改进以后可以预测多标签任务；办法就是使用了 logistics 函数，把数值转化为概率，设定阈值，比如把所有阈值大于 0.7 的类别都拿出来，这样就能实现多标签预测    

使用了残差连接  

核心网络架构 darknet53，没有池化和全连接层，全部是卷积层；下采样通过 stride=2 实现；3 种 scale 更多先验框  

13x13x3x85，这里的 85，网格坐标 4 个参数，confidence 1 个，80 个类别  


#### 代码 

用的是 PyTorch，数据是 COCO 数据集，19 个 G  

读数据用到了生成器 generator，因为全部读取 RAM 不够用，所以每次读取 64 张图片  

具体则是通过 \_\_get_item__ 一张一张图片读取的，先读图片名，再加上路径，路径用绝对路径，相对路径总是会有各种错误；长方形图片 padding 成正方形；再读标签  

Darknet 类，再 \_\_init__ 中，首先是 parse_model_config 读取 cfg 文件，然后是 create_modules 构建网络；route 层完成的就是拼接操作，原来的层和后面的层 concat 在一起，就是 Mask R-CNN 中的 C4 和 P5 的拼接操作；shortcut 层就是一个加法操作，不是拼接；YoloLayer 构建了 3 个最后的 YOLO层；    

然后是 forward 函数，往网络中传入数据；YOLO 层的 forward 函数，第一个的 shape 是 (4, 255, 15, 15), PyTorch 是 channel first，4 是 batch_size, 255 是 channel 数，15 是长宽，就是 grid 的长宽；prediction 就是最后要预测的结果，view 就是 reshape，num_samples 就是 batch_size 就是 4，num_anchors 就是每一个 scale 里面有 3 中大小不同的 anchor，num_classes 80，后面的 5 就是 4 个坐标值和 1 个 confidence；过了这一步以后 shape 就变成了 (4, 3, 15, 15, 85)；prediction 的前 4 个坐标就是 xywh，第 5 个是 confidence，再往后就是 cls 预测；compute_grid_offset 就是把相对于格子的坐标转化为绝对坐标，就是相对坐标再加上偏移量；build_target 计算损失，第一步就是把标签转化成和输入一样的 shape；4 个损失函数，第一个是位置误差，后面两个是 confidence 损失，在 obj confidence 中，obj 标签是 1，noobj 标签是 0. 在 noobj confidence 中，noobj 标签是 1，obj 标签是 0. noobj 还有一个 λ 权重参数调整重要程度。最后是 80 分类的分类误差。后面有很多内容，都是在做坐标转换；  

最后算 loss 值  


### YOLOv4  

v4 单 GPU 就可以跑起来  

Mosaic 数据增强，可以间接增加 batch_size，可以增强遮挡图片的识别  
DropBlock drop 掉一块儿区域，增加学习难度，减少过拟合风险  
Label Smoothing 就是是的 0.95，不是的 0.05，不要太绝对，留一些余地  
CIoU  IoU 的问题，如果 IoU 等于 0，就不能算梯度；GIoU 引入了一个封闭形状；DIoU 添加了距离；CIoU 考虑了重叠面积、中心点距离、长宽比  
DIoU-NMS 就是重叠的同时考虑了中心点的距离，距离大的就不要抑制，加了一个 DIoU；SOFT-NMS 重叠了降低分，看看后面变现满足就留下，不满足再去除  


SPPNet 就是用 maxpooling 使得最后的输出特征一致  
CSPNet 就是一部分再走卷积，另一部分直接拿过来 concat；可以减少计算量，速度更快  
SAM 是从 CBAM 转化过来的，CBAM 全称 Convolutional Block Attention Module，就是增加了 channel attention modul 和 spatial attention module，channel attention module 就是说比如有 256 个 channel，可以为 channel 加上权重值，spatial attention module 就是给图片的位置加上权重；SAM 就是只增加了 spatial attention module，而且取消了 maxpooling。  
PAN Path Aggregation Network 就是 FPN 又增加了 P2-P5 一共 4 层的向上的路径，而不是 C1 到 C5 的 100 多层的 resnet；论文里用的是 concat 而不是加法    
激活函数用了 Mish，而不是 relu，relu 太绝对了  













PyTorch YOLOv3 没有用虚拟环境，如果基础配置有了较大的更改，就重新配置虚拟环境  

python 3.6  
pytorch 1.6.0  

#### train.py  

epochs 改成 6  
batch_size 改成 1  

