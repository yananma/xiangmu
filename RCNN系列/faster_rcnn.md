
## Faster R-CNN  

Faster R-CNN 由两部分组成，Fast R-CNN 和 RPN；Faster R-CNN 只是加了一个 rpn 层，其他的和 Fast R-CNN 一样。Faster R-CNN 的核心就是 RPN，RPN 可以使用 GPU 来进行框的提取，解决了 Fast R-CNN 的瓶颈  

RPN 的两个 loss，一个判断是前景还是背景，一个是坐标回归  

用的是 VGG 作为 backbone  

锚框，3 种大小，3 种长宽，每个节点生成 9 个框  

在 training 的时候，越界的 anchors 就 ignore，在 testing 的时候，越界的就 clip  

training 中 ignore 以后剩 6000，再 NMS 阈值 0.7 剩 2000，再按 score 值排序取 topN，

#### 代码  

安装，在 GitHub 上找 Windows 版本，配置就是讲了一个怎么切换 Python 版本；遇到问题在 GitHub 的 issue 里面搜  

用的数据集是 PASCAL VOC 800 多兆，类别是 20 个，加上背景是 21 个；annotation 是标注  

从 if \_\_main__ 中的 train 进去，第一件事情就是初始化 VGG 网络  

然后就是取数据，拿到图片数据 imdb，5011 张，然后是 get_flipped_image，图片变为 10022 张，label 要改一下坐标；然后是读标签数据 roidb；RoIDataLayer 函数 shuffle 一下数据，有一个 output_dir 创建输出文件夹  

到这里，数据就处理完了，后面就是网络架构了 VGG 和 RPN  

从 create_architecture 函数进去，先是 3 种大小，3 种长宽，一共 9 种框；  

然后就是 bulid_network 函数，先读取预训练 weight

再在 build_network 函数中的进入 build_head 函数来搭建 VGG16，就是 slim repeat 函数搭积木；VGG 用的卷积是 3x3 的，不改变大小，只有 maxpooling 改变，4 个 pooling，16 倍  

然后是在 build_network 函数中的进入 build_rpn 函数来搭建 rpn 网络，build_rpn 函数里有一个 \_anchor_componet，是 anchor 相关，里面有一个 generate_anchor_pre 函数，这个 generate_anchor_pre 函数里有一个 generate_anchors 函数，画网格，生成框；出来 generate_anchors 函数，再回到 generate_anchor_pre 函数以后，再乘以 feat_stride 映射到原图上，到这里 \_anchor_componet 就算完了；回到 build_rpn 函数，\_anchor_componet 下面是 rpn 在特征图上的 3x3 卷积，然后是 2x9=18 个二分类，4x9=36 个框坐标回归  

然后是在 build_network 函数中的进入 build_proposals 函数，有一个 \_proposal_layer，里面有 proposal_layer，在这个 proposal_layer 里面做的就是筛选，先算 IoU，这个 IoU 是在原始图像上算的，＞0.7 的就是物体是前景，＜0.7 就不是物体是背景，2x9 的分类的 label 用的就是这个 IoU 的结果；对于框的回归，只有前景才用做，第一步算 IoU 就先筛选掉一批；然后是 NMS，NMS 再筛选掉一批；然后是是越界的 ignore；然后就剩下 200 多个了可能，再 topN 筛选，剩下 128 个。有一个 bbox_transform_inv 就是一个 dx dy 和论文里是一样的，和靠得最近的 ground truth 做回归；有一个越界的 clip，看看论文，看看代码，到底是 clip 还是 ignore；order 做了排序。build_proposals 函数里还有一个 \_anchor_target_layer，用来构建 rpn labels。build_proposals 函数里还有一个 \_proposal_target_layer，对一个物体来说，找到指定的 ground truth 是多少，就是一个打标签的过程，对 rpn 的标签进行制作。  


然后是在 build_network 函数中的进入 build_predictions 函数，就是最后的预测了，要连全连接层，fc layer。  

再往后就是指定损失函数，有 4 个损失函数  

![faster_rcnn](https://github.com/yananma/xiangmu/blob/master/RCNN%E7%B3%BB%E5%88%97/faster_rcnn.png)



python 3.5  
tensorflow 1.4.0  

时间是 2018 年 1 月 5 号前  
去 pypi 搜当时的版本 看 release   
