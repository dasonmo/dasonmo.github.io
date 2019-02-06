---
layout: post
title:  "ElasticSearch 用于简单搜索"
date:   2019-01-31
categories: elasticsearch
tags: elasticsearch LTR
comments: true
---

* content
{:toc}

简单的利用elasticsearch进行关键字搜索的例子





## ElasticSearch概述
1. 分布式实时文件存储，可将每一个字段存入索引，使其可以被检索到。
2. 实时分析的分布式搜索引擎。分布式：索引分拆成多个分片，每个分片可有零个或多个副本。集群中的每个数据节点都可承载一个或多个分片，并且协调和处理各种操作；负载再平衡和路由在大多数情况下自动完成。
3. 可以扩展到上百台服务器，处理PB级别的结构化或非结构化数据。也可以运行在单台PC上。
4. 支持插件机制，分词插件、同步插件、Hadoop插件、可视化插件等。

#### 总体方案
![pic_1_1]({{ "/asset/image/1_1.png" | absolute_url }})
1. 使用同一个elasticsearch库进行Faq（文档）检索和管理
2. 检索
    - 需要一个检索脚本。功能是将用户**query放入curl查询语句**中。curl请求后再进行**回包解析**。判断检索的第一位文档得分是否低于阈值，是则加入检索库；否则返回文档结果。
3. 管理
    - 需要一个管理脚本。功能包括**查询索引**，**增加索引**，**删除索引**，**查询文档**，**增加文档**，**更改文档**，**删除文档**。
4. 如何设置阈值
    - 建立测试索引，增加一批文档。
    - 编写一批与库中已有文档均不相似的测试用例，使用测试用例进行检索。
    - 编写脚本统计检索出的第一位文档的得分。
    - 根据得分设置阈值，低于该阈值加入检索库中。
    

#### 运行ElasticSearch
1. 解压elasticsearch压缩包到任意目录temp_path。
   tar -zxvf elasticsearch_6.2.4.tar.gz -C temp_path
2. 运行elasticsearch。
   temp_path/elasticsearch/bin/elasticsearch -d
3. 检查是否已运行，访问9200端口，若出现elasticsearch欢迎界面则成功运行。
   curl http://localhost:9200/?pretty

   ![pic_1_2]({{ "/asset/image/1_2.png" | absolute_url }})
   
#### 索引操作
1. 索引介绍
索引（index）是将若干个分区进行分组的逻辑空间，也就是文档存储的地址。
es中使用index API使数据可以被存储并检索。以下为索引的meta字段。

    | meta data | 解释  | 
    | --- | --- |
    | index | 文档存储的地方 |  
    | type | 文档代表的对象的类 |  
    | id | 文档的唯一标识 | 

2. 查询所有索引的情况
    curl 'localhost:9200/_cat/indices?v' 
    ![pic_1_3]({{ "/asset/image/1_3.png" | absolute_url }})
    
3. 新建Faq库索引，索引名称为faq
   ```json
    curl -XPUT 'localhost:9200/faq?pretty' -H "Content-Type: application/json" -d'{
    "settings" : 
    { 
        "index":
        {
            "analysis" : 
            { 
                "analyzer" : 
                { 
                    "seg_analyzer" : 
                    { 
                        "tokenizer" : "ik_max_word",
                        "filter": "unique"
                    }
                }
             }
        }
    },
    "mappings":
    {
        "faq":
        {
            "properties": 
            {
                "faq_id":
                {
                     "type": "long"
                },
                "standard_query": 
                { 
                     "type": "text",
                     "analyzer": "seg_analyzer",
                     "search_analyzer": "ik_smart"
                }
            }  
        }      
    }
    }'
    ```
    ![pic_1_4]({{ "/asset/image/1_4.png" | absolute_url }})
    
    
4. 单次往faq索引插入一条文档，文档的type也为faq，id为1
    ```shell    
    curl  -XPOST 'localhost:9200/faq/faq/1?pretty' -H "Content-Type: application/json"  -d'{
        "faq_id":1,
        "standard_query":"什么是固定汇率"
    }'
    ```
    ![pic_1_5]({{ "/asset/image/1_5.png" | absolute_url }})
    
5. 获取文档
    curl -XGET 'localhost:9200/faq/faq/1?pretty'
    ![e45d364079f5df461d9ef9f224c7d1bc.png](en-resource://database/8423:1) 
    
6. 批量索引操作 一次添加两个faq文档
    ```shell
    curl -XPOST 'localhost:9200/faq/faq/_bulk?pretty' -H "Content-Type:application/json" -d'
    {"index":{"_id":"2"}}
    {"faq_id": 2,"standard_query":"什么是逆回购"}
    {"index":{"_id":"3"}}
    {"faq_id": 3,"standard_query":"存款保险制度详解"}
    '
    ```
    ![pic_1_6]({{ "/asset/image/1_6.png" | absolute_url }})
    
#### 查询操作
1. 查询faq索引，name字段包含“逆回购”的文档，输出结果按_score值进行排序
    ```shell
    curl -XPOST 'localhost:9200/faq/_search?pretty' -H "Content-Type: application/json" -d'{
        "query":
        {
            "match": 
             {
                "standard_query":"逆回购"
             } 
        }
    }'
    ```
    ![pic_1_7]({{ "/asset/image/1_7.png" | absolute_url }})
    
2. 查看返回的首文档，根据_score值判断是否加入检索库。

#### 删除操作
1. 删除单个文档
    ```shell
    curl -XDELETE 'localhost:9200/faq/faq/1?pretty'
    ```
    ![pic_1_8]({{ "/asset/image/1_8.png" | absolute_url }})
2. 删除整个索引（慎重）
    ```shell
    curl -XDELETE 'localhost:9200/faq?pretty'
    ```

