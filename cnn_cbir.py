from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2

import os

# multiprocessing
from joblib import Parallel, delayed
import multiprocessing

# logging
from tqdm import tqdm
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def rmac(x, L=3, ovr=0.4, norm=True, eps=1e-6, padding=0):
#     ovr = 0.4 # desired overlap of neighboring regions
   
    # possible number regions for the longer dimension
    # written in this way, we can control the maximum number of siliding windows
#     possible_ms = np.array([1, 2, 3, 4, 5, 6, 7])
    N, C, H, W = x.size()
    w = np.minimum(W, H) # window size at scale 1
#     b = (np.maximum(H, W) - w)/possible_ms # step sizes for different possible number of regions

    # we want the number of region that makes the overlapping is as close as possilble to 0.4
    # steps[idx] regions for long dimension
    # (tmp, idx) = torch.min(torch.abs(((w**2 - w*b)/w**2)-ovr), 0)
#     idx = np.argmin(np.abs((1-b/w) - ovr))
    
#     m = possible_ms[idx]
    
    regions_ijww = []
    fea = []
    for l in range(1, L+1):
        wl = int(np.floor(2*w/(l+1))) # window size at scale l
        wl = np.maximum(wl, 2)
        sl = int(np.floor((1-ovr)*wl)) # stride size at scale l
        sl = np.maximum(sl, 1)
        pl= padding if padding is not None else sl
        xl = F.max_pool2d(x,
                          kernel_size=(wl, wl),
                          stride=(sl, sl),
                          padding=(pl, pl))   
        newh, neww = xl.size(2), xl.size(3)
        regions_ijww += [ (i*sl, j*sl, wl, wl) for i in range(newh) for j in range(neww)]
        fea.append(xl.view(N, C, -1))
    fea = torch.cat(fea, dim=2)
    fea = fea / (torch.norm(fea, p=2, dim=1, keepdim=True) + eps)
    fea = fea.transpose(1, 2) # (N, C, R ) -> (N, R, C)
    regions_ijww = np.array(regions_ijww)
    logging.debug('fea_size(): {:s}'.format(str(fea.size())))
    logging.debug('regions_ijww.shape: {:s}'.format(str(regions_ijww.shape)))
    assert fea.size(1) == regions_ijww.shape[0]
    return fea, regions_ijww


class RMAC(nn.Module):

    def __init__(self, L=3, over=0.4, norm=True, eps=1e-6, padding=0):
        super(RMAC, self).__init__()
        self.L = L
        self.over = over
        self.norm = norm
        self.eps = eps
        self.padding = padding

    def forward(self, x):
        return rmac(x, L=self.L,
                       over=self.over,
                       norm = self.norm,
                       eps = self.eps,
                       padding=self.padding)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'L=' + '{}'.format(self.L) + ')'

    

class FeatureExtractor:
    def __init__(self, backbone='vgg16', cache_dir='feature_cache', pool='rmac', eps=1e-6,
                 use_gpu=True):
        """
        args:
            backbone: the model used for feature extraction
        """
        assert isinstance(backbone, str), 'Callable object is not supported currently!'
        if backbone == 'vgg16':
            self.cnn = models.vgg16(pretrained=True)
            del self.cnn.classifier
        elif backbone == 'vgg19':
            self.cnn = models.vgg19(pretrained=True)
            del self.cnn.classifier
        else:
            raise NotImplementedError("Only 'vgg16' and 'vgg19' are suppported.")
            
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.eval()
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([ transforms.ToTensor(),
                                              normalize])
        
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.eps = eps
        self.pool = pool
        
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        self.cnn.to(self.device)
#         print('Using device: ', self.device)
#         logging('current device: {:s}'.format(self.device))
        
    def _map_im_path_to_cache_path(self, im_path):
        im_name = os.path.basename(im_path)
        cache_name = os.path.splitext(im_name)[0] + '.pth'
        cache_path = os.path.join(self.cache_dir, cache_name)
        return cache_path
    
    def get_im_feature_by_path(self, im_path, force_compute=False):
        cache_path = self._map_im_path_to_cache_path(im_path)
        if (not force_compute) and os.path.isfile(cache_path):
            logging.debug('cached feature for {:s} is found, directly loading it.'.format(im_path))
            fea_dict_im = torch.load(cache_path)
        else:
            logging.debug('computing feature for {:s}...'.format(im_path))
            im = cv2.imread(im_path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            fea_dict_im = self.compute_im_feature(im)
            torch.save(fea_dict_im, cache_path)
        return fea_dict_im
    
    def get_bb_mat(self, patches, im_path):
        bbs = []
        logging.debug('computing bounding boxes in {:s}'.format(im_path))
        for patch in patches:
            feat_dict_im = self.get_im_feature_by_path(im_path)
            feat_dict_patch = self.compute_im_feature(patch, pool='mac')
            logging.debug('feat_im.size(): {:s}'.format(str(feat_dict_im['reg_feat_mat'].size())))
            logging.debug('feat_patch.size(): {:s}'.format(str(feat_dict_patch['ag_feat_vec'].size())))
            similarity = torch.matmul(feat_dict_im['reg_feat_mat'],
                                      feat_dict_patch['ag_feat_vec'])
            logging.debug('similarity.size(): {:s}'.format(str(similarity.size())))
            val, ind = torch.max(similarity, dim=0)
            bb = feat_dict_im['regions_ijhw'][ind.item()]
            bbs.append(bb)
        return np.array(bbs)      
            
    def get_db_feature_matrix(self, im_paths, force_compute=False):
        cache_path = os.path.join(self.cache_dir, 'db_fea_mat.pth')
        if (not force_compute) and os.path.isfile(cache_path):
            logging.info('cached database feature matrix is found in {:s}, loading it directly.'.format(cache_path))
            db_fea_mat = torch.load(cache_path)
        else:
            logging.info('computing feature for {:d} images...'.format(len(im_paths)))
            db_fea_mat = []
            for im_path in tqdm(im_paths):
                fea_dict_im = self.get_im_feature_by_path(im_path, force_compute)
                db_fea_mat.append(fea_dict_im['ag_feat_vec'])
            db_fea_mat = torch.stack(db_fea_mat)
            torch.save(db_fea_mat, cache_path)
            logging.info('database feature matrix is computed and saved!')
        return db_fea_mat
    
    def compute_top_matches(self, im, db_fea_mat, top_k=50):
        im_fea_dict = self.compute_im_feature(im)
        similarity = torch.matmul(db_fea_mat, im_fea_dict['ag_feat_vec'])
        scores, inds = torch.topk(similarity, k=top_k)
        return scores, inds
        
    
    def compute_im_feature(self, im,
                           pool=None,
                           aggregation='sum'):
        '''
        extract raw regional feature matrix and aggregated feature vector.
        
        args:
            img: a CHW, RGB numpy array image
            pool: str,the final pooling method to get verctorized feature
            aggregation: str, the method used for feature aggregation
            cache_dir: direc.tory for caching features, if set to None, feature won't
                       be cached.
            im_name: file name for cached feature, note that, the suffix will be replaced
                     by '.pth'
            eps: float, eps for L2 normalization
            recompute: boolean, whether force to recompute instead of loading cached feature
            
        return:
            fea_dict: a dict {'fea_mat': fea_mat, 'regions_ijhw': regions_ijhw,
                     'aggr_fea': aggr_fea}, where fea_mat is a raw feature matrix
                     of shape (r, c). And regions_ijhw is a bounding box matrix of
                     shape (r, 4), r is the number of regions. Of course, r = 1 if using 
                     'mac' pololing. aggr_fea is aggregated feature tensor of shape (c, ).
        '''
        if pool is None:
            pool = self.pool
            
        assert pool in ['mac', 'rmac'], 'Only mac and rmac are supported.'
        assert aggregation in ['sum', 'ave'], 'Only sum and ave aggregation are supported.'
        
        with torch.no_grad():
            im_tensor = self.transform(im).unsqueeze(0).to(self.device)
            fea = self.cnn.features(im_tensor)
        
            _, c_im, h_im, w_im  = im_tensor.size()
            _, c_f, h_f, w_f = fea.size()
            s_h = h_im / h_f
            s_w = w_im / w_f

            if pool == 'rmac':
                reg_feat_mat, regions_ijhw = rmac(fea, L=3, ovr=0.5, padding=0, norm=True) # (1, R, C)

                # TODO: add PCA-whitening and another L2 normalization as post-processing here

                reg_feat_mat = reg_feat_mat.squeeze() # [1, r, c] -> [r, c]
                logging.debug('reg_feat_mat.size(): {:s}'.format(str(reg_feat_mat.size())))

                # back project bounding boxes
                regions_ijhw = regions_ijhw * np.array([[s_h, s_w, s_h, s_w ]])
                regions_ijhw = np.floor(regions_ijhw)

                # aggregate feature
                r = reg_feat_mat.size(0)
                ag_feat_vec = torch.sum(reg_feat_mat, dim=0, keepdim=False) # (r, c) -> (c,)
                logging.debug('ag_feat_vec.size(): {:s}'.format(str(ag_feat_vec.size())))
                if aggregation == 'sum':
                    ag_feat_vec = ag_feat_vec / (torch.norm(ag_feat_vec,
                                                            p=2,
                                                            dim=0,
                                                            keepdim=True) + self.eps)
                elif aggregation == 'ave':
                    ag_feat_vec = ag_feat_vec / r

                else:
                    raise NotImplementedError('Only sum and ave aggregation are supported.')

                return {'reg_feat_mat': reg_feat_mat, 
                        'regions_ijhw': regions_ijhw,
                        'ag_feat_vec': ag_feat_vec}

            elif pool == 'mac':
                feat_vec = F.max_pool2d(fea, (h_f, w_f)).squeeze() # (1, c, h_f, w_f) -> (c, )
                feat_vec = feat_vec / (torch.norm(feat_vec, p=2, dim=0, keepdim=True) + self.eps)
                return {'reg_feat_mat': feat_vec.unsqueeze(0),
                        'regions_ijhw': np.array([[0, 0, h_im, w_im]]),
                        'ag_feat_vec': feat_vec}
            else:
                raise NotImplementedError('Only mac and rmac are supported.')
        
    

class SearchEngine:
    def __init__(self, db_root, fea_extractor):
        
        self.db_root = db_root
        self.im_names = sorted(os.listdir(db_root))
        self.im_paths = [ os.path.join(db_root, im_name) for im_name in self.im_names ]
        
        self.fea_extractor = fea_extractor
        
        self.db_fea_mat = None
        
#         self.cache_dir = cache_dir
        
        # to speed up retrieval, we make feature matrix stick in memory
        # of course, this is not scalable
#         self.db_fea_mat = None       
    
    
    def build(self, force_compute=False):
        logging.info('building database feature matrix...')
        self.db_fea_mat = self.fea_extractor.get_db_feature_matrix(self.im_paths,
                                                                   force_compute)
#         if force_compute:
#             os.makedirs(cache_dir, exists_ok=True)
#             for im_path in tqdm(self.im_paths):
#                 im_name = os.path.basename(im_path)
#                 im = cv2.imread(im_path)[:, :, ::-1]
#                 im_fea = self.fea_extractor.compute_im_feature_by_path(im_path,
#                                                                        recompute=True)
#             self.db_fea_mat = self.fea_extractor.compute_db_fea_mat(self.im_paths,
#                                                                     recompute=True)
    
    
    def retrieve_img(self, img, top_k=50):
        '''
        args:
            img: CHW, RGB numpy array image
            top_k: int, top k images to retrieve
        return:
            result: a list of length top_k, each item is a (im_path, sim_score) tuple
        '''
#         query_fea, _ = self.fea_extractor.compute_feature(img,
#                                                           pool='rmac')
        
        scores, inds = self.fea_extractor.compute_top_matches(img,
                                                              self.db_fea_mat,
                                                              top_k=top_k)
        result = []
        for i in range(top_k):
            result.append((self.im_paths[inds[i]], scores[i].item()))
#         result = self.fea_extractor.get_top_matches(img,
#                                                     self.im_paths,
#                                                     top_k=top_k)
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
            logging.info('computing bounding box for retrieved {:d} images...'.format(top_k))
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
            bbs: a (n, 4) numpy array representing xyhw bounding boxes
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
    
    
