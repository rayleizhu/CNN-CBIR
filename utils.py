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


def draw_bbs_to_img(im, bbs=None, bb_format='xywh_mat', mode='show'):
    if bbs is None:
        im_with_bb = im
    else:
        assert bb_format in ['xywh_mat', 'corner_list', 'ijhw_mat']
        if bb_format == 'xywh_mat':
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
        elif bb_format == 'ijhw_mat':
            corners = []
            for row in bbs:
                x_l = row[1]
                x_r = row[1] + row[3]
                y_u = row[0]
                y_d = row[0] + row[2]
                corners.append(np.int32([[x_l, y_u], 
                                         [x_l, y_d],
                                         [x_r, y_d],
                                         [x_r, y_u]
                                        ]
                                       ).reshape(-1, 1, 2)
                              )
        else:
            corners = [ corner_mat.reshape(-1, 1, 2) for corner_mat in bbs ]

        im_with_bb = im
        for i, corner_mat in enumerate(corners):
            blue = int((1 - i/len(corners)) * 255)
            green = 255 - blue
            im_with_bb = cv2.polylines(im_with_bb, [corner_mat], True, (0, green, blue), 3, cv2.LINE_AA)
    assert mode in ['show', 'return']
    if mode == 'show':
        plt.figure()
        plt.imshow(im_with_bb)
    else:
        return im_with_bb
    
    
def get_row(items):
    row = ''
    for item in items:
        row += '| {:s} '.format(item)
    row += '|\n'
    return row

def generate_table(item_list, title=None, col_num=4):
    table = ''
    if title is None:
        title = [ 'item_{:d}'.format(i) for i in range(1, col_num+1) ]
    else:
        col_num = len(title)
    table += get_row(title)
    table += get_row(['-'] * col_num)
    for i in range(0, len(item_list), col_num):
        table += get_row(item_list[i:i+col_num])
    return table

    
    