
debug 没有成功，视频都看完以后，再回头过一遍，每个项目都点开看一看，读一读，看看哪些项目没有上传代码，就是要 debug 的  

每一个都试一试，成了就成了，不成就算了  

## 《数学基础》
#### 1、汽车价格回归分析  
模型算法：线性回归  
missingo画图查看缺失值、特征相关性分析(相关性高的特征去掉)及画图、 
<br>

## 《机器学习实战》
#### 1、信用卡欺诈检测  
模型算法：LogisticRegression  
类别个数柱形图、下采样、混淆矩阵、  
软件版本过旧，运行出错。
<br>

#### 2、基于随机森林的气温预测
>模型算法：随机森林 
>
>#### 1-随机森林
>处理时间数据、graphviz 画决策树、特征重要性、 
>
>#### 2-数据与特征对随机森林的影响
>seaborn pairplot、特征重要性累加、
>
>#### 3-随机森林参数选择 
>随机搜索 RandomizedSearchCV、网格搜索 GridSearchCV、

#### 3、贝叶斯新闻分类任务
模型算法：朴素贝叶斯  
jieba 分词、停用词、wordcloud、TF-IDF 提取关键词、词袋模型、
<br>

#### 4、支持向量机
模型算法：支持向量机  
SVM 画图、核函数、调参数松弛因子 C 和高斯核函数参数 σ、seaborn heatmap 画混淆矩阵、
<br>

#### 5、xgboost 保险赔偿业务 
>模型算法：xgboost 
> 
>#### part1_data_discovery 
>pandas 处理数据、画图查看数据、
> 
>#### part2_xgboost 
>xgboost 参数含义、xgboost 调参、

#### 6、科比生涯数据分析 
模型算法：这个项目的目的主要是演示怎么处理数据  
pandas 处理数据、画图、
<br>

#### 7、贷款申请项目
模型算法：LogisticRegression、随机森林  
有一段算混淆矩阵的代码、
<br>

## 《机器学习进阶》
#### 1、数据特征
>#### 数值特征
>离散值处理、LabelEncoder、One-hot Encoding、get_dummies、binning、分位数切分、对数变换、Timestap、时间相关特征
>
>#### 文本特征
>nltk 停词、词袋模型、N-Grams 模型、TF-IDF、LDA、word2vec、

#### 2、梯度提升算法GBDT 
模型算法：GBDT、Xgboost、LightGBM  
三个算法的简单原理、算法参数、效果性能对比、
<br>

#### 3、饭店流量预测 
模型算法：LightGBM  
完全是 pandas 应用教学，从这个例子学会了 groupby、apply、inplace、shift、merge、log 变换、
<br> 

#### 4、人口普查数据分析
模型算法：KNN、Logistic Regression、Random Forest、Naive Bayes、Decision Tree、Gradient Boosting Trees<br>
这个项目用了 6 个算法，最后通过比较 AUC 来选取最好的模型。<br>
特征清洗和特征工程流程和常用方法、特征选择、特征相关性、特征重要性、单变量分析、双变量分析、ROC、AUC、
<br>

#### 5、线性判别分析 LDA 
模型算法：LDA  
这个项目是用代码实现了 LDA 的算法公式，演示了整个算法
<br>

#### 6、高斯混合模型 GMM
模型算法：PCA、KMeans、GMM    
<br>

#### 7、隐马尔可夫模型 HMM
模型算法：HMM  
hmmlearn 工具包用法、维特比算法求解、<br>
<br>

#### 8、NLP 自然语言处理方法对比  
模型算法：词袋模型、word2vec、深度学习 CNN 和 RNN  
正则表达式过滤特殊字符、画词频图并截断频数少的、

#### 9、使用word2vec实现分类任务
模型算法：LogisticRegression、词袋模型、word2vec  
word2vec 参数、语言数据处理流程：去除 HTML 标签、去除标点、分词、去除停用词、重新组合句子、

#### 10、数据处理与特征工程
>模型算法：Linear Regression、Support Vector Machine Regression、Random Forest Regression、Gradient Boosting Regression、K-Nearest Neighbors Regression 
>
>#### 数据预处理
>可以设置图像统一尺寸、缺失值处理(自己定义的函数)、EDA、去除离群点、kdeplot、相关性分析、
>
>#### 建模
>imputer 缺失值填充、交叉验证、
>
>#### 分析
>按重要性进行特征选择、lime 展示特征影响、


确实是有需要，要看函数和变量的，才用 GPU，一般不用。  

没什么收获，耽误很多时间  

## 《TensorFlow2 实战》

#### 1、神经网络进行气温预测  
Keras 建模  


#### 2、神经网络分类任务 
tf.data 的常用函数、fashion mnist训练、画图显示结果(条形图显示结果概率)、  


#### 3、读取网络模型
加载训练保存的模型、  


#### 4、猫狗识别
数据增强、迁移学习、callback、  


#### 5、图像增强实例
旋转、平移、缩放、channel_shift、翻转、rescale、填充、  


#### 6、迁移学习实战
TFRecords、


#### 7、递归神经网络与词向量模型
word2vec原理、CBOW、Skip-gram、负采样、  


#### 8、RNN 文本分类任务
embedding、加载 glove 模型、BiLSTM、  


#### 9、TextCNN 文本分类
TextCNN、


#### 10、时间序列预测
两层 LSTM、单点预测、区间预测、  


#### 11、DCGAN
代码实现算法  



## 《Keras项目实战》 


#### 1、股票数据预测
点预测、区间预测、保存网络结构图片、


#### 2、文本分类实战  
词袋模型、LogisticRegression 基础模型、全连接网络、Embedding、word2vec、LSTM、CNN、RandomizedSearchCV、  


#### 3、seq2seq网络实战
核心是 library 文件夹下的 seq2seq.py 文件，要动手 debug 一遍、  


## 《PyTorch框架实战》
PyTorch 是重点  

#### 1、分类回归任务
分类任务 MNIST 回归任务气温预测  


#### 2、图像识别核心模块
数据增强、花分类、


#### 3、迁移学习
和上一个项目是在一起的，是一个项目分了两大节内容、加载 ResNet 预训练模型、


#### 4、新闻数据集文本分类实战
LSTM、TextCNN、再看一遍，看老师 debug 的变量内容，比自己 debug 好  


#### 5、对抗生成网络 GAN


#### 6、CyCleGAN
GAN 的核心是看损失函数  


#### 7、基于3D卷积的视频分析与动作识别
跑不动，看视频就行了  


#### 8、bert 
看一遍视频，包括前面的理论部分。bert 还是应该看 TensorFlow 版本的  


#### 9、PyTorch框架实战模板解读
看  

## 《OpenCV计算机视觉实战》
学这么课的关键是看 cv2.后面的函数  

#### 1、图像基本操作
读图像、读视频、截部分内容、提取颜色通道、边界填充、数值计算、图像融合、  


#### 2、阈值与平滑处理
灰度图、HSV、图像阈值、图像平滑、  


#### 3、图像形态学操作
腐蚀操作、膨胀操作、开运算与闭运算、梯度运算、礼帽与黑帽、  


#### 4、图像梯度计算
Sobel 算子、Scharr 算子、laplacian 算子、  


#### 5、边缘检测
Canny 边缘检测、高斯滤波器、梯度和方向、非极大值抑制、双阈值检测、  


#### 6、图像金字塔与轮廓检测
高斯金字塔、拉普拉斯金字塔、绘制轮廓、轮廓特征、轮廓近似、模板匹配、匹配多个对象、  


#### 7、直方图与傅里叶变换
直方图均衡化、自适应直方图均衡化、滤波、  


#### 8、信用卡识别数字识别


#### 9、文档扫描OCR识别


#### 10、Harris
harris 角点检测、  


#### 11、SIFT
图像尺度空间、多分辨率金字塔、高斯差分金字塔、关键点的精确定位、特征点的主方向、生成特征描述、  


#### 12、全景图像拼接
特征匹配、单应性矩阵、  


#### 13、停车场车位识别



#### 14、答题卡识别判卷



##### 15、背景建模
帧差法、混合高斯模型去噪、  


#### 16、光流估计
Lucas-Kanade 算法、


#### 17、疲劳检测

1、《图像操作》  
图像基本操作  
图像处理  


## 已读论文
AlexNet、VGG、GoogLeNet、ResNet  

R-CNN、Fast R-CNN、Faster R-CNN、Mask R-CNN  

YOLO、YOLOv2、YOLOv3、YOLOv4  

SSD  

Attention、Transformer、Bert  

Batch Normalzation   



## 待完成项目

《物体检测实战》  
6、Mask R-CNN 源码  
24、人体姿态识别 demo  


《自然语言处理实战》  
8、NLTK 工具包  
12、spacy 工具包  
16、结巴分词器  
37、商品信息可视化  
134、LSTM 情感分析  
139、NLP 相似度模型(读)  
146、seq2seq 网络架构  
148、seq2seq 基本模型  
150、对话机器人  
173、LSTM 时间序列分析  


《bert》  
13、源码  
55、医学糖尿病数据集  


《语音识别技术》  
1、seq2seq 序列网络模型  
8、LSA 语音识别模型实战  
17、stargancv2 变声器论文原理  
24、stargancv2 变声器源码实战  
35、语音分离 ConvTasnet 模型  
41、ConvTasnet 语音分离实战  
49、语音合成技术概述 和后面 tacotron  


《对抗生成网络 GAN》  
4、损失函数解释说明  
8、CycleGAN  
18、StarGAN  
54、图形分辨率重构实战  
63、基于 GAN 的图像补全实战  


《物体检测 YOLO 系列》  
8、YOLO 整体思想和网络架构  
13、YOLOv2 改进细节详解  
28、YOLOv3 源码解读  



《机器学习实训营》  
21、评估方法  
29、线性回归实验分析  
45、逻辑回归(多分类)  
57、逻辑回归实验  
69、kmeans 代码实现  
75、聚类算法实验分析  
94、决策树代码实现  
101、决策树实验分析  
109、集成算法实验分析  
131、支持向量机实验分析  
149、神经网络代码实现  
167、贝叶斯代码实现  
173、关联规则实战  
192、代码实现word2vec词向量(读)  
205、打造音乐推荐系统  


《数据分析机器学习实战集锦》  
2、Python 实战关联规则  
9、爱彼迎数据集分析与建模  
20、基于相似度的酒店推荐系统  
26、商品销售额回归分析  
35、绝地求生数据集探索分析  
65、NLP 核心模型 word2vec (看)  
78、数据特征预处理  
91、银行客户还款可能性预测  
104、图像特征聚类分析实战  


《数据挖掘竞赛 优胜解决方案实战》  
1、快手短视频用户活跃度分析  
11、工业化工生产预测  
18、智慧城市道路通行时间预测  
27、特征工程建模可解释工具包  
36、医学糖尿病数据命名实体识别  
43、贷款平台风控模型 特征工程  
51、新闻关键词抽取模型  
69、用电敏感客户分类  
77、机器学习项目实战模板  


《模型部署与剪枝优化》  
1、PyTorch 框架部署实战  
7、YOLOv3 物体检测部署实例  
11、docker 实例演示  
18、TensorFlow serving 实战  
23、模型剪枝 Network Slimming 算法  
34、MobileNet 网络模型架构  




