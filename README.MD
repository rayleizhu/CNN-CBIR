



## 1. Results

### 1.1 image retrieval and object localization for No. 1 - 5 query

The following table has 11 rows, 5 columns, each column is for one query. In each column, the first one is query image, followed with ten retrieved ones.
You can click image to see big pictures in `result` directory

| Q1 | Q2 | Q3 | Q4 | Q5 |
| - | - | - | - | - |
| ![](result/Q1/match_top_0.png) | ![](result/Q2/match_top_0.png) | ![](result/Q3/match_top_0.png) | ![](result/Q4/match_top_0.png) | ![](result/Q5/match_top_0.png) |
| ![](result/Q1/match_top_1.png) | ![](result/Q2/match_top_1.png) | ![](result/Q3/match_top_1.png) | ![](result/Q4/match_top_1.png) | ![](result/Q5/match_top_1.png) |
| ![](result/Q1/match_top_2.png) | ![](result/Q2/match_top_2.png) | ![](result/Q3/match_top_2.png) | ![](result/Q4/match_top_2.png) | ![](result/Q5/match_top_2.png) |
| ![](result/Q1/match_top_3.png) | ![](result/Q2/match_top_3.png) | ![](result/Q3/match_top_3.png) | ![](result/Q4/match_top_3.png) | ![](result/Q5/match_top_3.png) |
| ![](result/Q1/match_top_4.png) | ![](result/Q2/match_top_4.png) | ![](result/Q3/match_top_4.png) | ![](result/Q4/match_top_4.png) | ![](result/Q5/match_top_4.png) |
| ![](result/Q1/match_top_5.png) | ![](result/Q2/match_top_5.png) | ![](result/Q3/match_top_5.png) | ![](result/Q4/match_top_5.png) | ![](result/Q5/match_top_5.png) |
| ![](result/Q1/match_top_6.png) | ![](result/Q2/match_top_6.png) | ![](result/Q3/match_top_6.png) | ![](result/Q4/match_top_6.png) | ![](result/Q5/match_top_6.png) |
| ![](result/Q1/match_top_7.png) | ![](result/Q2/match_top_7.png) | ![](result/Q3/match_top_7.png) | ![](result/Q4/match_top_7.png) | ![](result/Q5/match_top_7.png) |
| ![](result/Q1/match_top_8.png) | ![](result/Q2/match_top_8.png) | ![](result/Q3/match_top_8.png) | ![](result/Q4/match_top_8.png) | ![](result/Q5/match_top_8.png) |
| ![](result/Q1/match_top_9.png) | ![](result/Q2/match_top_9.png) | ![](result/Q3/match_top_9.png) | ![](result/Q4/match_top_9.png) | ![](result/Q5/match_top_9.png) |
| ![](result/Q1/match_top_10.png) | ![](result/Q2/match_top_10.png) | ![](result/Q3/match_top_10.png) | ![](result/Q4/match_top_10.png) | ![](result/Q5/match_top_10.png) |

### 1.2 quantitative results on validation data

TBA.


## 2. How to use

Go to the [demo.ipynb]() for a step-by-step tutorial. 
If you want to further develop based on this repository, you may refer to section 4:  design, implementation, features, discussion.


## 3. Methodology

In the instance search task, given a image with one or multiple object bounding box(es), you are required to retrive images containing similar objects and moreover, locate them. In this report, I implemented two methods:  CNN based and SIFT-BOW feature based. 

Basically, we use [VGG16](https://arxiv.org/pdf/1409.1556.pdf) + [RMAC (regional maximum activation of convolutions)](https://arxiv.org/pdf/1511.05879.pdf) layer to get feature representation for retrieval and object localization.

![VGG16 architechture](assets/VGG16-architecture-16.png "VGG16 architechture")

### 3.1 pipeline

#### preprocessing
We first mask image with provided bounding boxes, that is, we replace pixels outside object bouding boxes with zeros. This is to make feature extraction focus on target query region.

#### feature extraction
We use VGG16 and RMAC layer to get image embedding.

For VGG 16, **the classifier part (fully connected layers) are cut thus not used**. We start from extracting feature map in pool5 layer. since the fully connected layer is gone, the remaining fully convolutional part can take as input of any shape. Then we pass pool5 feature into RMAC layer to get aggregated multiscale feature as image embedding vector.

The mechanism of RMAC operation is simple:
1. uniformly generate a set of multiscale sliding windows, each determines a region. Assume we have $R$ regions.
2. for $i_{th}$ region, apply max-pooling to get a 512-dimensional feature vector, then apply L2 normalization to get  $v_{i}$. Thus we have $R$ feature vectors $\{ v_{i} \}_{i=1}^R$.
3. aggregate all regional feature vectors by summation $\sum\{ v_{i} \}_{i=1}^R$, to get final feature vector $v_{ag}$.

#### image retrieval
With aggregated feature extracted, we can evluate cosine similarity between query image and database images. Top ranked images are returned.

#### object localization (optional)
We localize image by computing cosine similarity vgg16 pool5  **global max-pooling feature** of query object region and $\{ v_{i} \}_{i=1}^R$ in retrieved images in last step. Assume $v_m, m \in [R]$ is the most similar embedding, we then back project the region corresponding to $v_m$ to input image using linear mapping.


### 3.2 diffrences from original RMAC

#### we do not use approximate integral max-pooling

The approximate integral max-pooling operation described in [paper](https://arxiv.org/pdf/1511.05879.pdf) can reduce time complexity for RMAC caculation from $O(W^2 \dot H^2)$ to $O(W \dot H)$, using the [integral image trick](https://en.wikipedia.org/wiki/Summed-area_table). 

![approximate integral max-pooling](assets/approximated_max_pooling.png)

This is of course quite a gain **for serialized scenario**. However, since computing integeral image is actually a dynamic prgramming strategy which involves with heavy data dependency, it's **Not friently to parallel computing**, thus useless even harmful if you use GPU for inference.

#### we don't apply approximate max-pooling localization refinement

Of course I didn't implement **approximate** integral max-pooling, as depicted above. But I even DO NOT apply max-pooling localization refinement. The main reason is that, I don't think it's worthy for the cost it brings, and moreover, I'm lazy. 

#### no extra PCA-whitening  and L2-normalization step to post-process regional features

I don't comprehend the motivation of these two steps. Then I just keep it simple :).
![extra pca whithening and L2 normalization](assets/pca_l2.png)

#### no re-ranking and query expansion

Again, I just want to keep my system simple, so such extensions are not included. But maybe I will add it later.
![re-rank and qe](assets/rerank_qe.png)


### 3.3 Analysis


#### What's the strength of RMAC?
* Objects have have varied size in different images, RMAC feature takes into account the effect of scale, thus makes retrieval more precise.
* Another strength is that, rough object localization is redily applicable by searching the most similar region in retrieved image

#### Why don't I using RPN or anchors for object localization?
RPN or anchors used in object detection methods are trained to localize limited class of objects. However, in image retrieval, possible objects emerged in query image can be anything, in another word, instance retrieval can be regarded as examplar based open-set object detection. RPN and anchor are not scalable to such cases.



## 4.  Design, implementation, features, discussion

TBA


## 5.  Reference

* [VGGNet](https://arxiv.org/pdf/1409.1556.pdf)
* [RMAC layer](https://arxiv.org/pdf/1511.05879.pdf)
* [integral image / Summed-area table](https://en.wikipedia.org/wiki/Summed-area_table))


## 6. TODO

* add dynamical indexing function
* unify SIFT-BOW/SIFT-TFIDF to the this framework.
* query expansion and reranking

