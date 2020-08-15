3.2、3.3 不用读，

第一次使用 CNN 在这么大的数据集上  

使用 GPU  

使用 ReLU 作为激活函数，ReLU 的速度要快得多。  

使用 Data augmentation、Dropout 防止过拟合  

详细介绍了 Dropout：是一个集成模型；因为可能被 Drop 掉，所以每一个 neuron 必须要变得 robust；  

Dropout 在防止过拟合上效果很好  


模型结构是五层卷积，三层全连接(看d2l)  

Conv  
Pool  

Conv  
Pool  

Conv  
Conv  
Conv  
Pool  

Dense  
Dense  
Dense  

