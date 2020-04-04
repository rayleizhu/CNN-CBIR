import logging
logger = logging.getLogger('__SearchEngine__')

class SearchEngine:
    def __init__(self, im_paths, fea_extractor):
        
        # self.db_root = db_root
        # self.im_names = sorted(os.listdir(db_root))
        # self.im_paths = [ os.path.join(db_root, im_name) for im_name in self.im_names ]
        self.im_paths = im_paths
        
        self.fea_extractor = fea_extractor
        
        # to speed up retrieval, we make feature matrix stick in memory
        # of course, this is not scalable
        self.db_fea_mat = None       
    
    
    def build(self, force_compute=False):
        logger.info('building database feature matrix...')
        self.db_fea_mat = self.fea_extractor.get_db_feature_matrix(self.im_paths,
                                                                   force_compute)
    
    
    def retrieve_img(self, img, top_k=50):
        '''
        args:
            img: CHW, RGB numpy array image
            top_k: int, top k images to retrieve
        return:
            result: a list of length top_k, each item is a (im_path, sim_score) tuple
        '''
        scores, inds = self.fea_extractor.compute_top_matches(img,
                                                              self.db_fea_mat,
                                                              top_k=top_k)
        result = []
        for i in range(top_k):
            result.append((self.im_paths[inds[i]], scores[i].item()))
            
        return result
    
    
    def retrieve_object(self, img, bbs, top_k=10, locate=True):
        '''
        restrieve images in database containing similar objects, and
        locate them if argument 'locate' is set to True
        
        args:
            img: CHW, RGB numpy array image
            bbs: a (n, 4) numpy array representing xyhw bounding boxes
            top_k: int, number of images you want to retrieve
            locate: boolean, if set to True, rough object location wil be returned
        return:
            result: a list of length top_k, each item is a (im_path, sim_score, bb_mat) tuple
            if 'locate' is set to True, else (im_path, sim_score) tuple
        '''
        # we mask query image with object mask before retrieval
        masked_img, patches = self._get_masked_img(img, bbs)
        top_k_img = self.retrieve_img(masked_img, top_k=top_k)
        
        if locate:
            logger.info('computing bounding box for retrieved {:d} images...'.format(top_k))
            result = []
            for img_path, score in tqdm(top_k_img):
                bb_mat = self.fea_extractor.get_bb_mat(patches, img_path)
                result.append((img_path, score, bb_mat))
            return result
        
        return top_k_img
        
      
    def _get_masked_img(self, img, bbs):
        '''
        helper function for creating bounding box masked image and 
        patches containing single objects
        
        args:
            img: CHW, RGB numpy array image
            bbs: a (n, 4) numpy array representing xywh bounding boxes
        return:
            masked_img: image with region outside bounding boxes masked by zeros
            patches: list of n CHW, RGB patches containing single object
        '''
        patches = []
        masked = np.zeros_like(img)
        for bb in bbs:
            x_l = bb[0]
            x_r = bb[0] + bb[2]
            y_u = bb[1]
            y_d = bb[1] + bb[3]
            patches.append(img[y_u:y_d, x_l:x_r])
            masked[y_u:y_d, x_l:x_r] = img[y_u:y_d, x_l:x_r]
        return masked, patches

