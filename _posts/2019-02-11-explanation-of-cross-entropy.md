---
layout: post
title: "The Sense of Cross Entropy"
date: 2019-02-11
categories: Machine Learning
tags: Cross-Entropy
comments: true
mathjax: true
---

* content
{:toc}

交叉熵主要用于度量两个概率分布间的差异性信息，在机器学习领域应用广泛。在自然语言处理中，语言模型的性能就主要使用交叉熵进行表征。因此，深刻理解交叉熵的意义可以增进我们对机器学习各个领域的认识






## 定义
对于特定的集合，两个分布$p$和$q$的交叉熵定义如下：
\begin{equation}
H(p,q) = E_p[-\log{}q]
\end{equation}
其中$E_p$表示在概率分布$p$上的数学期望  

本文主要讨论离散概率分布；对于离散概率分布$p$和$q$而言，交叉熵定义如下：
\begin{equation}
H(p,q) = - \sum_{x_i} p(x)\log{}q(x)
\end{equation}

## 公式理解
信息论当中的[Kraft–McMillan theorem](https://en.wikipedia.org/wiki/Kraft%E2%80%93McMillan_inequality)表明：任何一套编码方案，它的目的是对一段信息进行编码，得到的编码可以从一个可能性集合$X$中确定一个数值$x_i$。这样的编码方案可以看做是$X$的一个概率分布，表示为$q(x_i)=2^{l_i}$,其中$l_i$是确定$x_i$所需要的编码位长度（比特）。

因此从信息论而言，交叉熵可解释为在真实分布为$P$，观测认为的有偏差的分布为$Q$的情况下，对于每个元素所需要的最短编码长度。由于真实分布为$P$，所以公式取概率分布$P$上的数学期望。

当$P=Q$时，也就是预测分布与真实分布相等时，所需编码长度最小，此时交叉熵最小。

## 语言模型中的交叉熵

实际中存在分布未知而需要测量交叉熵的场景，例如语言模型。语言模型在训练集$T$上得到，然后在测试集上测量其交叉熵，目的是测量语言模型在预测时的准确度。假设真实分布为$p$（此分布必须在所有的语料库成立），而当前语言模型给出的有偏差的分布为$q$，则可用下式表示交叉熵：
\begin{equation}
H(T,q) = - \sum_{i=1}^{N} \frac{1}{N}\log_{2}q(x_i)
\end{equation}  

其中$N$是测试集的大小，$q(x)$是从训练集学习到的$x$发生的概率。

Wiki上认为此公式是对真实交叉熵的蒙特卡罗估计，训练集是从真实分布$p(x)$采样而来的。  

从形式上，是否可以认为此公式是使用$1/N$代替真实分布，也就是假设语料库的真实分布为均匀分布，从而计算两个分布之间的交叉熵呢？

## KL散度
加入KL散度后，交叉熵可表达为：
\begin{equation}
H(p,q) = E_p[-\log{}q] = H(p) + D_{KL}(p||q)
\end{equation}  
其中$H(p)$为分布$p$的信息熵,$D_{KL}(p||q)$定义如下

\begin{equation}
D_{KL}(p||q) = \sum_{i}p(i)\log{}\frac{p(i)}{q(i)}
\end{equation}  

在机器学习领域，KL散度又称为相对熵。$D_{KL}(p\|\|q)$可以理解为使用分布$q$替代分布$p$时获得的信息增益。

当比较两个分布的相似程度时，KL散度（相对熵）和交叉熵本质上是等价的（只相差一个常数$H(p)$，$p$是固定的真实分布），当$p=q$时，KL散度为0，交叉熵为$H(p)$。

## Reference
1. [Cross Entropy Wiki](https://en.wikipedia.org/wiki/Cross_entropy)