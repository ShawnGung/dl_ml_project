# 周杰伦歌词生成
如何利用循环神经网络GRU生成周杰伦的歌词
数据集 : 周杰伦从第一张专辑《Jay》到第十张专辑《跨时代》中的歌词

## 总结
- 利用相邻采样更好的保证hidden输入保存尽量多前面序列的信息
- 在训练上,注意单纯的view只是按顺序来改变tensor的形状,要根据实际情况来利用transpose和view + continugous来获取我们想要的数据形状
- 裁剪梯度,循环神经网络中训练,很容易出现梯度爆炸或者衰减的情况。面对梯度爆炸,我们需要进行梯度裁剪
- 我们通常使用困惑度（perplexity）来评价语言模型的好坏。




## Refereneces
http://zh.d2l.ai/chapter_recurrent-neural-networks/lang-model-dataset.html  
