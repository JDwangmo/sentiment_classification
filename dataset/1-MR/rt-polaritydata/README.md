Movie Review Data set
============

Introduction

数据集来源：
http://www.cs.cornell.edu/people/pabo/movie-review-data 

任务说明：
用户电影评论情感极性分类，二分类 （ positive / negative ）

=======

Reference

Bo Pang and Lillian Lee. 
Seeing stars: Exploiting class relationships for sentiment categorization
with respect to rating scales. 
In Proceedings of the ACL, 2005.


=======

State-Of-the-Art

模型名                  10折验证准确率           来源
CNN-rand	              76.1                Kim, EMNLP 2014
CNN-static	              81                  Kim, EMNLP 2014
CNN-non-static            81.5                Kim, EMNLP 2014
CNN-multichannel	      81.1                Kim, EMNLP 2014

[Kim, EMNLP 2014 ] : Yoon Kim, Convolutional Neural-Networks for 
Sentence Classification. EMNLP 2014.


=======

Data Format Summary 

- 全部utf8 encoding
  Specifically: 
  * rt-polarity.pos.utf8 ： 5331 positive 样例
  * rt-polarity.neg.utf8 ： 5331 negative 样例
   
=======

Label Decision 

positive : 情感正性，表示接受或者开心（pleasant）等
negative : 情感负性，表示不接受或者不开心（unpleasant）等




