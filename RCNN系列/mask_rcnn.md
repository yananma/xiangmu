
### Mask R-CNN 

Faster R-CNN 早一些，用的是 VGG，Mask R-CNN 用的是 ResNet  

#### 代码 

原来 rcnn 系列都是只用最后一层特征图，比如有 5 层，就只用第 5 层；后面的特征图包含整体的信息，但是检测小目标效果不好。所以提出了一个网络叫做 FPN，就是利用多个阶段的特征图提取特征。大小相同的特征是一个阶段的特征。每次用的是同一个阶段特征图的最后一层。C1-C5 大小不同，channel 也不同，用 1x1 卷积得到 P2-P5，使得 channel 数相同，都是 256，C4 在和 P5 融合的时候，C4 也要用 1x1 的卷积变成 256 个 channel. 融合就是加起来，P5 和 C4 融合，P5 要上采样。  

class MaskRCNN 的 build 函数。对图像大小有限制，先做一个判断。input_image 就是输入图像 x，if model == "training" 里面是先读进来 5 个标签。因为特征图的大小不一样，会有数值上的差异，为了消除这个的影响，坐标都做了归一化。  

第一个断点就是 resnet_graph，得到了 5 个阶段的特征图。P5 是 C5 用 256 个 channel 的 1x1 卷积后得到的，后面就是 P5 upsampling，C4 1x1 改变 channel 数，再加起来融合。P2-P5 往右到 predict 还有一个 3x3 的卷积。这里是用 FPN 的方式完成的 rpn 的功能，就是要达到 region proposal 的目的。  

generate_pyramid_anchors 候选框的生成，遍历所有阶段特征图的所有位置，生成大小比例不同的候选框，存成 list。在不同的特征图上做候选框的时候，注意要映射到原始图像上。遍历 scale，就是在不同的特征图上按照不同的大小的比例生成框。进入 generate_anchors，框最开始都是按比例生成的，后面长宽还会调整，会再拉长拉宽一点。meshgrid 画坐标棋盘的意思。一个坐标点上是生成 3 个框。要再映射到原始图像上。这里就是把全部的框全部都拿到手了，后面再筛选。  

build_rpn_model 做的和 Faster R-CNN 是一样的，差别就是在不同阶段的特征图上做 rpn，都是经过 512 个 channel 的 3x3 的卷积，到 prediction。这个 3x3 卷积，应该和 Faster R-CNN 在特征图上那个 3x3 卷积是一样的，这个 prediction 应该就是 2k 个前景背景分类和 4k 个坐标回归。  

代码上 build_rpn_model 函数，input_feature_map 就是输入，就是 P2-P5 经过 3x3 的卷积到 prediction，所以输入都是 256 个channel 的，就是大小根据阶段不同变化。进入 rpn_graph，首先是 shared，这个很简单，就是那个 3x3 卷积，所有的阶段往 prediction 用的是相同的 3x3 卷积，所以叫 shared。每一个 feature_map 来的时候都会经过这个 3x3 的卷积。得到 2k 个 anchor score 值，就是前景背景分类，再经过 softmax 得到概率值，再得到 4k 个回归值。  

再往下遍历每个阶段的特征图，得到的结果 append 起来。

ProposalLayer 先把 20 多万个框按照得分高低排序，再取前 6000 个做框的回归，再 NMS 取 2000 个。就是先做过滤，再把回归值都用上。

代码，先取 score 值，只管前景得分，背景的分就不管了，delta 是偏移量。topk 6000 个，apply_box_delta_graph 函数就是做的框的微调。clip_boxes_graph 越界的部分 clip 掉。再 NMS。最后得到的就是 rpn_roi。  

DetectionTargetLayer 主要是做了 8 大步看视频：padding 0 的去掉；一个框多个物体的去掉；判断前景背景；设置负样本是正样本的 3 倍；给每一个正样本指定类别，就是 IoU 最大的类别；算每一个正样本关于 ground truth 的 box 偏移量；为每一个正样本以最近的 ground truth 的 mask 为 mask；返回结果，其中负样本的 box 偏移量和 mask label 都是 0.代码做的就是这 8 大步。代码做的就是这个，有一个小的细节是算了所有的 roi 和 gt 的比较，算出每一个 roi 的 IoU，原来对这一点不是很清晰。   

roi 都是原图上的，要再映射到 feature map 上，就要用到 ROIAlign，双线性插值法。原来的 roi pooling 会有偏差，不准确，用插值法就可以用小数了。  

经过 PyramidROIAlign 函数以后，不管原来是从多大的特征图上来的，都变成了 7x7 统一大小。再加两层全连接。最后就是 mask。mask 这里是 pooling 成 14x14 的，和前面都是 7x7 不同。

最后是 5 个 loss。







#### balloon.py
208 行，epochs 改成 1  
88 行 STEPS_PER_EPOCH = 5  
卡在 epoch 1/30 不动，去 GitHub 的 issue 里搜，set use_multiprocessing=False and workers=1 in model.py  


Miniconda3-4.3.30.2  
python 3.6.3  
TensorFlow 1.4.0  
Keras 2.0.8  


训练配置参数  
train  
--dataset=../../balloon  
--weights=coco  

有了问题在 issue 里查  


