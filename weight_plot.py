import torch
import numpy as np
import os 
import matplotlib.pyplot as plt

model_name = 'ETTh2_40_5 _DLinear_ETTh2_ftM_sl50_ll20_pl25_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_1'
for root, dirs, files in os.walk("informer_checkpoints"):
    for name in files:
        model_path = os.path.join(root, name)
        if model_name not in model_path:
            continue
        weights = torch.load(model_path,map_location=torch.device('cpu'))
        weights_list = {}
        weights_list['seasonal'] = weights['Linear_Seasonal.weight'].numpy()
        weights_list['trend'] = weights['Linear_Trend.weight'].numpy()

        save_root = 'weights_plot'
        if not os.path.exists('weights_plot'):
            os.mkdir('weights_plot')
        if not os.path.exists(save_root):
            os.mkdir(save_root)

        for w_name,weight in weights_list.items():
            fig,ax=plt.subplots()
            im=ax.imshow(weight,cmap='plasma_r')
            fig.colorbar(im,pad=0.03)
            plt.savefig(os.path.join(save_root,w_name + '.jpg'),dpi=500)
            plt.close()
 