
GoogLeNet 太复杂了，用的不多，ResNet 以后，基本上就全都用 ResNet 了，所以还是以 ResNet 为主。  

要想有好的模型，就要增加模型的 size，一个是深度，一个是宽度。但是也会因此而增加参数数量，造成过拟合，还会增加计算量和存储消耗。  

一个解决办法就是使用 sparse 模型。  

核心就是 Inception 模块。  

第 4 节讲了结构的细节，为什么这么设计。其实还是试出来的。  

1x1 卷积是为了减少通道数量。  

GoogLeNet 是用 Inception 模块堆叠起来的。一共 22 层。  

证明 sparse structure 是可行的。  



理解 Inception，整篇论文就理解了 90%。其实没什么要理解的，知道结构就行。然后看 Figure3 和 Table1，结合 d2l 知道整体结构就完全掌握了。  

