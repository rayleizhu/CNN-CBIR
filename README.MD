# CBIR using VGG-RMAC feature


## 1. Results

### 1.1 image retrieval and object localization for No. 1 - 5 query (top_k = 3)

The following table has 4 rows, 5 columns, each column is for one query. **In each column, the first one is query image, followed with 3 retrieved ones.**
You can click images to see full size image in `result` directory.

For more results (top_k=10), Please go to [Demo.ipynb](https://github.com/rayleizhu/CNN-CBIR/blob/master/Demo.ipynb).

| Q1 | Q2 | Q3 | Q4 | Q5 |
| - | - | - | - | - |
| ![](result/Q1/match_top_0.png) | ![](result/Q2/match_top_0.png) | ![](result/Q3/match_top_0.png) | ![](result/Q4/match_top_0.png) | ![](result/Q5/match_top_0.png) |
| ![](result/Q1/match_top_1.png) | ![](result/Q2/match_top_1.png) | ![](result/Q3/match_top_1.png) | ![](result/Q4/match_top_1.png) | ![](result/Q5/match_top_1.png) |
| ![](result/Q1/match_top_2.png) | ![](result/Q2/match_top_2.png) | ![](result/Q3/match_top_2.png) | ![](result/Q4/match_top_2.png) | ![](result/Q5/match_top_2.png) |
| ![](result/Q1/match_top_3.png) | ![](result/Q2/match_top_3.png) | ![](result/Q3/match_top_3.png) | ![](result/Q4/match_top_3.png) | ![](result/Q5/match_top_3.png) |

### 1.2 quantitative results on validation data

TBA.


## 2. How to use

Go to the [Demo.ipynb](https://github.com/rayleizhu/CNN-CBIR/blob/master/Demo.ipynb) for a step-by-step tutorial. 
If you want to further develop based on this repository, you may refer to section 4:  design, implementation, features and discussion.


## 3. Methodology

* You can refer to the methodology introduction [here](https://github.com/rayleizhu/CNN-CBIR/blob/master/docs/methodology.MD).

* But you are highly recommended to view it on my blog [Instance Search with CNN](http://rayleizhu.com/?p=457), where Latex syntax and UML flowchart is supported. 



## 4.  Design, implementation, features, discussion

* The two core componets are class SearchEngine() and FeatureExtractor() defined in cnn_cbir.py. 

* Writing a SIFT-BOW and SIFT-TFIDF feature extrator following the same contract of current CNN feature extractor, we can have a unified implementaion

* Feature caching is used to speed up inference. Feature embedding is computed on demand, so dynamically indexing new image on-the-fly is achievable (but not added yet, coming soon).

* When doing object localization, we apply level supression, since I observed that otherwise the localization will prefer larger window. see line xx in xx.

* The RMAC implementaion is minimal and more efficient than existing implementations. Moreover, I providde other option on pooling method ('RAAC') and regional aggregation (average, not tested).

* Experiments show that Level 1 pooling is important for retrieval. If initial scale is 2, the performance get worse (see dev notebook)



## 5. TODO

* add table of contents in README.md
* add dynamical indexing function
* unify SIFT-BOW/SIFT-TFIDF to the this framework.
* query expansion and reranking

