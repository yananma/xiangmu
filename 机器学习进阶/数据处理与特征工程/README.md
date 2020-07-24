
### 数据预处理 
有一个报错，是两个小错误，一个是 log 中出现了零，这个错误已经解决，在 log 括号中加一个很小的数，比如 1e-2  
一个是传入的数字，有无效的，这个没有解决  
这个结果在后面没有调用，所以运行到这一步后跳过，运行下面  


#### 建模
这个程序运行了两个小时才运行完，主要是交叉验证。  

导包错误：from sklearn.preprocessing import Imputer, MinMaxScaler  
改成：from sklearn.impute import SimpleImputer as Imputer  
&emsp;&emsp;&emsp;  from sklearn.preprocessing import MinMaxScaler  


#### 分析
导包错误：from sklearn.preprocessing import Imputer, MinMaxScaler  
改成：from sklearn.impute import SimpleImputer as Imputer  
&emsp;&emsp;&emsp;  from sklearn.preprocessing import MinMaxScaler  


