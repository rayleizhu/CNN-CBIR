from torch.utils.data import DataLoader, Dataset
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImgDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.img_names = sorted(os.listdir(data_root))
        self.transform = transform
        
    def __getitem__(self, idx):
        path = os.path.join(self.data_root, self.img_names[idx])
        img = Image.open(path)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, self.img_names[idx]
    
    def __len__(self):
        return len(self.img_names)
    
    
def load_image_and_bbs(im_path, bb_path):
    img = cv2.imread(im_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bb_df = pd.read_csv(bb_path, sep=' ', header=None, index_col=False)
    bb_mat = bb_df.to_numpy()
    return img, bb_mat


def show_img_with_bbs(im, bbs, bb_format='xyhw_mat'):
    assert bb_format in ['xyhw_mat', 'corner_list']
    if bb_format == 'xyhw_mat':
        corners = []
        for row in bbs:
            x_l = row[0]
            x_r = row[0] + row[2]
            y_u = row[1]
            y_d = row[1] + row[3]
            corners.append(np.int32([[x_l, y_u], 
                                     [x_l, y_d],
                                     [x_r, y_d],
                                     [x_r, y_u]
                                    ]
                                   ).reshape(-1, 1, 2)
                          )
    else:
        corners = [ corner_mat.reshape(-1, 1, 2) for corner_mat in bbs ]
    im_with_bb = cv2.polylines(im, corners, True, (0,0,255), 3, cv2.LINE_AA)
    plt.figure()
    plt.imshow(im_with_bb)
    
    
    
    