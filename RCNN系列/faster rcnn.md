
查看是怎么生成锚框的：network.py 的 \_anchor_componet 函数，第四行 generate_anchor_pre 点进去看，到 snippets.py 第一行，generate_anchors 点进去跳到 generate_anchors.py，添加 pysnooper，可以直接运行这个文件，会报错，不用管， 会出结果，读完就懂了。

关于 proposal 在 train 里没法调用，可能在 test 里才能用，暂时先略过  


python 3.5  
tensorflow 1.4.0  

时间是 2018 年 1 月 5 号前  
去 pypi 搜当时的版本 看 release   
