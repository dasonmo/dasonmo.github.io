---
layout: post
title: "An Introduction from RankNet to LambdaMart "
date: 2019-02-08
categories: LTR
tags: LTR RankNet LambdaMart
comments: true
mathjax: true
---

* content
{:toc}

搜索排序经典算法剖析


## 问答(QA)系统架构简介
#### 功能模块
1. Index
  - 用于存储QA对的检索库
2. Matching Models
  - 用于给QA对进行打分的模型，可以包括W2v/N-gram/TF-IDF/BM25/Rnn/Cnn等模型，各类模型对每一个QA对输出一个打分。可以理解为Rank阶段的特征工程。
3. Ranking Model
  - 排序模型，用于特征融合，多采用树模型或神经网络模型。实际上是一个回归问题。

![pic_2_1]({{ "/asset/image/2_1.png" | absolute_url }})

#### 检索过程
1. 由于检索库的QA对数量庞大，需要先使用轻量的检索工具返回一定数量的QA对，然后送入Matching models中进行打分。
2. Matching models进行打分，构造特征向量。
3. Ranking model对向量化的多个QA对进行排序，输出得分最高的QA对作为答案。

## Ranking Model类型
> 以下称每个QA对为一个doc  

1. Point-wise  
   将每个doc作为一个instance，采用分类或回归的方法进行训练；缺点是忽略了doc之间的排序关系
2. Pair-wise  
   将每两个doc作为一个instance，考虑其偏序关系
3. List-wise  
   将doclist整体作为一个instance进行训练

## 训练数据构造
1. 给定Matching Models
2. 对某一query进行检索将得到有序的doclist。此doclist可按照某重要特征进行粗排序
3. 对当前doclist进行人工标注。label表示doc和query的相关程度，数值越大越相关
![pic_2_2]({{ "/asset/image/2_2.png" | absolute_url }})

## RankNet
> RankNet是一种Pairwise的ranking model，使用神经网络进行学习。RankNet由微软研究院的Chris Burges等人在2005年ICML上的一篇论文Learning to Rank Using Gradient Descent中提出。  

#### 定义  
\begin{align}
&S_{ij}\in\{0,\pm1\}  \\\
1&: doc_i比doc_j更相关  \\\
0&: 同样相关  \\\
-1&：doc_j比doc_i更相关  
\end{align}

假设RankNet训练的任意时刻，模型对$doc_i$和$doc_j$的打分分别为：    
\begin{align}
&S_i=f(x_i) \\\
&S_j=f(x_j) \\\
x_i和x_j&分别为各自doc的特征向量  
\end{align}

模型认为$doc_i$比$doc_j$更相关的概率为  
\begin{equation}
P_{ij}\equiv P(U_i\triangleright U_j)\equiv \frac{1}{1+e^{-\sigma(S_i-S_j)}}
\end{equation}  
实际$doc_i$比$doc_j$更相关的概率为  
\begin{align}
\bar{P_{ij}} &= \frac{1}{2}(1+S_{ij}) \\\
1&:doc_i比doc_j更相关\\\
\frac{1}{2}&:同样相关\\\
0&:doc_j比doc_i更相关
\end{align}  

#### 交叉熵损失函数
\begin{equation}
C = -\bar{P_{ij}}\log_{}P_{ij}-(1-\bar{P_{ij}})\log_{}(1-P_{ij})
\end{equation}
代入$P_{ij}$和$\bar{P_{ij}}$:
\begin{equation}
C = \frac{1}{2}(1-S_ij)\sigma(s_i-s_j)+\log(1+e^{-\sigma(s_i-s_j)})
\end{equation}
>即使两个doc的得分相同（模型初始化时），若label不同，此时损失$c=\log_{}2$,依然会根据label分别向上下调整排序

#### 求导得到梯度
\begin{align}
\frac{\partial{C}}{\partial{s_i}} = \sigma(\frac{1}{2}(1-S_{ij})-\frac{1}{1+e^{\sigma(s_i-s_j)}}) = -\frac{\partial{C}}{\partial{s_j}}
\end{align}
梯度大小相等，方向相反

#### 更新网络参数
\begin{equation}
w_k \to w_k - \eta\frac{\partial{C}}{\partial{w_k}} = w_k - \eta(\frac{\partial{C}}{\partial{s_i}}\frac{\partial{s_i}}{\partial{w_k}} + \frac{\partial{C}}{\partial{s_j}}\frac{\partial{s_j}}{\partial{w_k}})
\end{equation}

>以上为$doc_i$和$doc_j$这一doc pair的参数更新过程，而这样的pair有$n+(n-1)+...+1$对。更新一次参数开销显然过大，能否直接得到$doc_i$的梯度，减少更新次数从而加快训练速度呢？

#### 改写求导公式
\begin{align}
\frac{\partial{C}}{\partial{w_k}} &= \frac{\partial{C}}{\partial{s_i}}\frac{\partial{s_i}}{\partial{w_k}} + \frac{\partial{C}}{\partial{s_j}}\frac{\partial{s_j}}{\partial{w_k}} = \sigma(\frac{1}{2}(1-S_{ij})-\frac{1}{1+e^{\sigma(s_i-s_j)}})(\frac{\partial{s_i}}{\partial{w_k}} - \frac{\partial{s_j}}{\partial{w_k}}) \\\
&= \lambda_{ij}(\frac{\partial{s_i}}{\partial{w_k}} - \frac{\partial{s_j}}{\partial{w_k}})
\end{align}
定义
\begin{equation}
\lambda_{ij} \equiv \frac{\partial{C}(s_i-s_j)}{s_i} = \sigma(\frac{1}{2}(1-S_{ij})-\frac{1}{1+e^{\sigma(s_i-s_j)}})
\end{equation}

$w$参数的改变量：