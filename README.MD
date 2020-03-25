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

Step-by-step tutorial: [Demo.ipynb](https://github.com/rayleizhu/CNN-CBIR/blob/master/Demo.ipynb)  

Dataset(google drive): [pg_data](https://drive.google.com/file/d/1Owlw7OdzjcCcE8MHWzPdVuPcy1ndlW2u/view) | [supplementary](https://drive.google.com/file/d/1swUy1dcfPQnoTLWUT7S_rwPnFVFENPYW/view)

If you want to further develop based on this repository, you may refer to section 4 (not finished yet):  design, implementation, features and discussion.


## 3. Methodology

* Methodology introduction: [[github](https://github.com/rayleizhu/CNN-CBIR/blob/master/docs/methodology.MD)] | [[my blog](http://rayleizhu.com/?p=457)] (recomended, for better view of mathematical notations and flowchart)


## 4. Design, implementation, discussion

### 4.1 design
![](assets/class.svg)

#### SearchEngine

* build()
  * get db_feature_mat computed or loaded to memory.
  
* retrieve_img()
  * image-level content retrieval.
  
* retrieve_object()
  * object-level content retrieval.similar to image-level retrieval, but will preprocess image by masking * image with object bounding boxes and locate objects on top ranked images (if needed).
  
* index_new_img()[pending]
  * index new images on the fly

#### FeatureExtractor

* compute_im_feature()
  * compute image-level feature embedding
  
* compute_top_matches()
  * compute top ranked image to retreieve
  
* compute_bb_mat()
  * locate objects in retrieved image

* \_map_im_path_to_cache_path()
  * map image path string to corresponding cached feature path string
  
* get_im_feature_by_img_path()
   * read cached image feature if found, otherwise compute and then cache it
   
* get_db_feature_matrix()
   * read database feature matrix if found, otherwise compute and ten cache it

**Currently, the implementation is slightly different from the design:**
* Function names
* BOWFeatureExtractor is not implemented yet, coming soon. [SIFT-BOW-CIBR](https://github.com/rayleizhu/SIFT-BOW-CBIR)

### 4.2 implementation

* When doing object localization, we apply level supression, since I observed that otherwise the localization will prefer larger window. see line xx in xx.

* The RMAC implementaion is minimal and more efficient than existing implementations. Moreover, I providde other option on pooling method ('RAAC') and regional aggregation (average, not tested).

* Experiments show that Level 1 pooling is important for retrieval. If initial scale is 2, the performance get worse (see dev notebook)

### 4.3 features

- [x] support index new image on the fly
- [x] support automatic caching
- [x] easy to extend framework


## 5. TODO

* add table of contents in README.md
* add dynamical indexing function
* unify SIFT-BOW/SIFT-TFIDF to the this framework.
* query expansion and reranking

