
读前面两节划线部分即可，很多内容已经过时了  

神经网络相对于传统机器学习的优势：不用手动去做特征工程  
全连接网络的缺点：参数过多，导致存储和计算要求过高，且容易过拟合；打乱了图像结构  
卷积神经网络的特点：局部连接、参数共享、池化操作  
为什么要 pooling：图像的平移不变性；减少参数  
<br>
<br>
LeNet5 结构图  
![GitHub](https://github.com/yananma/xiangmu/blob/master/%E8%AE%BA%E6%96%87/LeNet5/LeNet5.png)  

Conv  
Pool   
<br>
Conv  
Pool  
<br> 
Dense  
Dense  
Dense  
