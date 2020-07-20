
### 图像特征
pip install scikit-image  
不过在 import 的时候，还是用 skimage  
<br>

### 文本特征  

执行程序的代码下载不行，要手动下载。  
参考：https://blog.csdn.net/qq_38929464/article/details/104740271  

删除 nltk.download()  
添加：nltl.data.path  
创建 nltk_data/corpora 文件夹  
从 https://github.com/nltk/nltk_data 下载压缩包，解压，复制到文件夹下  
<br> 
主题模型 LDA 没有 n_topics 参数了，改用 n_components  

后面还有一个软件过期了，最后一段  
