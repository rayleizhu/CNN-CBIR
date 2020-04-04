from .feature_extractor_base import FeatureExtractor
from sklearn import cluster, neighbors, feature_extraction
import numpy as np
import cv2

import copy
import random

# multiprocessing
import joblib
from joblib import Parallel, delayed
import multiprocessing

# logging
from tqdm import tqdm
import logging

logger = logging.getLogger('__BoWFeatureExtractor__')
SEED = 10086

random.seed(SEED)
np.random.seed(SEED)


# to enable pickling cv2.KeyPoint object
# https://stackoverflow.com/a/48832618
import copyreg
def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)
copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)



class BoWExtractor(FeatureExtractor):
    def __init__(self, cache_dir='BoW_cache', im_feature='TF', pt_feature='SIFT',
                 vsize=1024, n_jobs=-1, top_k=50, metric='cosine'):
        super(BoWExtractor, self).__init__(cache_dir)
        assert im_feature in ['TF', 'TFIDF']
        assert pt_feature in ['SIFT'], 'Currently only support SIFT feature'
        self.im_feature = im_feature
        self.pt_feature = pt_feature
        self.vsize = vsize
        self.n_jobs = n_jobs
        self.top_k = top_k
        self.metric = metric

        self.kmeans = None
        self.nn_searcher = None

        if pt_feature == 'SIFT':
            self.pt_fea_extractor = cv2.xfeatures2d.SIFT_create()
        else:
            raise NotImplementedError(f"Pont feature extraction method {pt_feature} is not implemented!")


        if self.im_feature == 'TFIDF':
            self.tfidf_transformer = sklearn.feature_extraction.text.TfidfTransformer(norm='l1')
    

    def _get_kmeans(self, im_paths, force_compute=False, cluster_im_num=0.1):
        cache_path = os.path.join(self.cache_dir, 'kmeans.pth')
        if (not force_compute) and os.path.isfile(cache_path):
            logger.info(f'cached kmeans quantizer is found in {cache_path}, '\
                         'loading it directly.')
            kmeans = joblib.load(cache_path)
        else:
            # get image paths for point-level feature clustering
            assert cluster_im_num > 0.
            cluster_im_paths = copy.deepcopy(im_paths)
            random.shuffle(cluster_im_paths)
            if cluster_im_num <= 1.0:
                cluster_im_num = int(cluster_im_num * self.num_imgs)
            else:
                cluster_im_num = int(cluster_im_num)
            cluster_im_paths = cluster_im_paths[:cluster_im_num]

            logger.info(f'building vocabulary using {cluster_im_num:d} images...')
            # extract point-level feature matrices
            cluster_des_list = Parallel(n_jobs=self.n_jobs, backend='threading')\
                                        (delayed(self._aux_func_pt)(im_path)
                                         for im_path in tqdm(cluster_im_paths)
                                        )
            des_mat_all = np.concatenate(cluster_des_list, axis=0)
            logger.info(f'{len(des_mat_all):d} key points have been extracted!')


            # do clustering
            logger.info(f'running kmeans clustering with {self.vsize:d} centers...')
            # we use MiniBatchKMeans since it can handle large datasets
            kmeans = cluster.MiniBatchKMeans(n_clusters=self.vsize,
                                             init_size=10*self.vsize,
                                             batch_size=self.vsize,
                                             random_state=SEED)
            # TODO: try partial_fit()
            # https://www.programcreek.com/python/example/103492/sklearn.cluster.MiniBatchKMeans
            # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans.partial_fit
            kmeans.fit(des_mat_all)
            joblib.dump(kmeans, cache_path)
            logger.info(f'kmeans is built and cached to {cache_path}!')
        return kmeans


    def get_db_feature_matrix(self, im_paths, force_compute=False):
        cache_path = os.path.join(self.cache_dir, 'db_fea_nn_seacher.pth')
        if (not force_compute) and os.path.isfile(cache_path):
            logger.info(f'cached nearest neighbor seacher is found in {cache_path}, '\
                         'loading it directly.')
            db_fea_nn_seacher = joblib.load(cache_path)
        else:
            logger.info('getting kmeans clusterer...')
            self.kmeans = self.get_kmeans(im_paths)

            logger.info(f'Building {len(im_paths):d}*{self.vsize:d} image-level feature matrix...')
            im_fea_mat = Parallel(n_jobs=self.n_jobs, backend='threading')\
                                  (delayed(self._aux_func_im)(im_path, 'TF')
                                   for im_path in tqdm(im_paths)
                                  )
            im_fea_mat = np.stack(im_fea_mat, axis=0)
            logger.debug(f'self.im_fea_mat.shape: {im_fea_mat.shape}')
            
            if self.im_feature == 'TFIDF':
                im_fea_mat = self.tfidf_transformer.fit_transform(im_fea_mat)
                for i, img_path in enumerate(img_paths):
                    cache_path = self._map_im_path_to_cache_path(im_path)
                    fea_dict_im = joblib.load(cache_path)
                    fea_dict_im['TFIDF'] = im_fea_mat[i]
                    joblib.dump(fea_dict_im, cache_path)
            

            logger.info('building nearest neighbor searcher for image retrieval...')
            # to search most similar images, here we use sklearn's NN
            db_fea_nn_searcher = neighbors.NearestNeighbors(n_neighbors=self.top_k,
                                                           n_jobs=self.n_jobs,
                                                           metric=self.metric)
            db_fea_nn_searcher.fit(self.im_fea_mat)
            joblib.dump(db_fea_nn_seacher, cache_path)
            logger.info(f'nearest neighbor searcher is built and cached to {cache_path}!')
        return db_fea_nn_seacher


    def migrate_to_tfidf(self, im_paths):
        assert self.im_feature == 'TF', \
               f'current image level feature is {self.im_feature}'
        self.im_feature = 'TFIDF'

        db_fea_nn_searcher = self.get_db_feature_matrix(im_paths,
                                  force_compute=True)
        return db_fea_nn_searcher



    def _compute_pt_feature(self, img_bgr):
        '''
        args:
            img: cv2 BGR image
        return:
            (kp, des): kp is a list containing n feature point coordinate (cv2.Keypoint object),
                       des is a matrix of shape (n, 128)
        '''
        im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self.pt_fea_extractor.detectAndCompute(im_gray, None)
        return kp, des

    
    def _aux_func_pt(self, im_path):
        # this function is created to speed up point_level feature extraction using joblib
        # see self.build_vocabulary()
        im_bgr = cv2.imread(im_path)
        kp, des = self._compute_pt_feature(im_bgr)
        return des

    def _aux_func_im(self, im_path, feature=None):
        if feature is None:
            feature = self.im_feature
        im_fea_dict =  self.get_im_feature_by_path(im_path)
        return im_fea_dict[feature]


    def compute_im_feature(self, im_bgr, feature='TF'):
        assert feature in ['TF', 'TFIDF']

        kp, des = self._compute_pt_feature(im_bgr)
        word_idx = self.kmeans.predict(pt_feature_mat)
        hist = np.bincount(word_idx.ravel(), menlength=self.vsize)

        ret_dict = {'kp': kp, 'des': des, 'TF': hist}

        if feature == 'TFIDF':
            hist_idf = self.tfidf_transformer.fit_transform(np.expand_dims(hist, axis=0))
            hist_idf = hist_idf[0]
            ret_dict['TFIDF'] = hist_idf

        return ret_dict


    def get_im_feature_by_path(self, im_path, feature=None, force_compute=False):
        cache_path = self._map_im_path_to_cache_path(im_path)
        if (not force_compute) and os.path.isfile(cache_path):
            logging.debug('cached feature for {:s} is found, directly loading it.'.format(im_path))
            fea_dict_im = joblib.load(cache_path)
        else:
            logging.debug('computing feature for {:s}...'.format(im_path))
            im = cv2.imread(im_path)
            if feature is None:
                feature = self.im_feature
            fea_dict_im = self.compute_im_feature(im, feature)
            joblib.dump(fea_dict_im, cache_path)
        return fea_dict_im

    
    def compute_top_matches(self, im, db_fea_nn_searcher,
                            top_k=50):
        if top_k is None:
            top_k = self.top_k
        im_fea_dict = self.compute_im_feature(im)
        dists, inds = db_fea_nn_searcher.kneighbors(im_fea_dict[self.im_feature],
                                                    top_k,
                                                    return_distance=True)
        return dists, inds


    def get_bb_mat(self, patches, im_path, min_match_num=20):
        bbs = []

        if self.pt_feature in ['sift', 'surf']:
            norm = cv2.NORM_L2
        else: # ORB, BRIEF, BRISK
            norm = cv2.NORM_HAMMING
        bf = cv2.BFMatcher(norm, crossCheck=True)

        for patch in patches:
            feat_dict_im = self.get_im_feature_by_path(im_path)
            feat_dict_patch = self.compute_im_feature(patch)
            matches = bf.match(fea_dict_patch['des'],
                               fea_dict_im['des'])
            matches = sorted(matches, key = lambda x:x.distance)
            match_num = len(matches)
            if match_num < min_match_num:
                logger.warn(f'Only {match_num:d} matches have been found, '\
                            f'lower than match_num_for_hom {min_match_num:d}.')
            good_matches = matches[:min_match_num]
            kp_obj = feat_dict_patch['kp']
            kp_sim = feat_dict_im['kp']
            src_pts = np.float32([ kp_obj[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp_sim[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # homograpy wrapping for corner points
            h, w = patch.shape[:2]
            src_corners = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst_corners = cv2.perspectiveTransform(src_corners, M)
            bbs.append(dst_corners)

        return bbs