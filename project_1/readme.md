<!--
 * @Description: 
 * @Author: 唐健峰
 * @Date: 2023-09-14 15:21:44
 * @LastEditors: ${author}
 * @LastEditTime: 2023-09-15 16:06:57
-->
## 词袋模型
词袋模型(Bag of Words，BoW):将文本视为词汇表中的词语集合，然后计算每个词语在文本中的出现频率。这将创建一个词频向量，其中每个元素表示对应词汇表中词语的出现次数。词袋模型忽略了词语的顺序和语法结构，只关注词汇表中的词汇。

## 我采用的词袋模型的得分算法

$$f(x) = \frac{ TF(t, d) \cdot LDF(t, D)}{ e ^ { \left( -\sum_{i=0}^{N} p(x) \cdot \log_2(p(x)) \right)}}$$
