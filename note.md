
### Faster R-CNN  

Faster R-CNN 由两部分组成，Fast R-CNN 和 RPN  

Faster R-CNN 的核心就是 RPN，RPN 可以使用 GPU 来进行框的提取，解决了 Fast R-CNN 的瓶颈  

RPN 的两个 loss，一个判断是前景还是背景，一个是坐标回归  

用的是 VGG 作为 backbone  





