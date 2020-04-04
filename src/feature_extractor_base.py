from abc import ABC, abstractmethod
import os

class FeatureExtractor(ABC):
    def __init__(self, cache_dir='feature_cache'):
        '''
        args:
            cache_dir: the directory to cache extracted features
        '''
        super(FeatureExtractor, self).__init__()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _map_im_path_to_cache_path(self, im_path):
        '''
        e.g.: 'database_dir/0001.png' will be mapped to 'cache_dir/0001.pth'
        '''
        im_name = os.path.basename(im_path)
        cache_name = os.path.splitext(im_name)[0] + '.pth'
        cache_path = os.path.join(self.cache_dir, cache_name)
        return cache_path
    
    @abstractmethod
    def get_im_feature_by_path(self, im_path, force_compute=False):
        '''
        args:
            im_path: image path for a database image
            force_compute: bool, if set to true, the image feature will be forced to be remomputed
        return:
            image_feature: image_feature
        '''
        pass
    
    @abstractmethod
    def get_bb_mat(self, patches, im_path):
        '''
        get object bounding boxes on retrieved image which is stored in im_path
        '''
        pass
    
    
    @abstractmethod        
    def get_db_feature_matrix(self, im_paths, force_compute=False):
        '''
        get db_feature_matrix of shape (n_images_in_database, feature_length)
        pass
        '''
    
    @abstractmethod
    def compute_top_matches(self, im, db_fea_mat, top_k=50):
        '''
        args:
            db_feat_mat: database feature matrix of shape (n_images_in_database, feature_length)
            top_k: number of images to be retirieved
        return:
            scores: scores of top_k retrieved images, (top_k, ) float array
            inds: indices of top_k retrieved images, (top_k, ) int array
        '''
        pass
        
    
    @abstractmethod
    def compute_im_feature(self, im):
        pass
    