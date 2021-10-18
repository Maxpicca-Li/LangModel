# LangModel

基于人民日报标注语料（见附件`./训练语料utf-8.txt`），训练一个Bigram语言模型，并预测任意给定语句的语言概率。

## 说明

- 模型基础：Bigram
- 平滑方法：加一平滑、katz平滑

## 参考资料

- hanlp读取语料：[使用 HanLP 统计二元语法中的频次 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/187560424)
- katz原理：[把GoodTuring结合katz回退的平滑算法彻底说清楚 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/100256789)
- katz参考实现：[Bigram/katz.py at master · Neesky/Bigram (github.com)](https://github.com/Neesky/Bigram/blob/master/katz.py)

