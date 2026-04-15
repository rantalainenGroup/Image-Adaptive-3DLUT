#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
from utils.visualize_lut import vis_lut, vis_lut_ax
from models_x import *  
from matplotlib import cm
from matplotlib.colors import Normalize
from skimage.color import rgb2hsv
import PIL
from matplotlib.lines import Line2D
import torchvision  


# In[2]:


original_path = '/mnt/ssd/bojing/Image-Adaptive-3DLUT/data/Batch_1/tiles/'

# luts_path = '/mnt/ssd/ferbue/Image-Adaptive-3DLUT/LUTs/unpaired/exp_20/saved_models/LUTs_best_fid.pth'
# eval_path = '/mnt/ssd/ferbue/Image-Adaptive-3DLUT/LUTs/unpaired/exp_20/evaluation/best_fid_png/'
# test_df = pd.read_csv('/mnt/ssd/bojing/Image-Adaptive-3DLUT/dataframes/scanb_malmo_philips_xr_test.csv')
# SOURCE_DOMAIN = 'PHILIPS'

# EXP_ID = '20'
# SOURCE_DOMAIN = 'PHILIPS'
# TARGET_DOMAIN = 'XR'
# test_df = pd.read_csv('/mnt/ssd/bojing/Image-Adaptive-3DLUT/dataframes/scanb_malmo_philips_xr_test.csv')

# EXP_ID = '21'
# SOURCE_DOMAIN = 'S360'
# TARGET_DOMAIN = 'XR'
# test_df = pd.read_csv('/mnt/ssd/ferbue/Image-Adaptive-3DLUT/dataframes/scanb_malmo_s360_xr_test.csv')

EXP_ID = '22'
SOURCE_DOMAIN = 'APERIO'
TARGET_DOMAIN = 'XR'
test_df = pd.read_csv('/mnt/ssd/ferbue/Image-Adaptive-3DLUT/dataframes/scanb_malmo_aperio_xr_test.csv')

base_path = '/mnt/ssd/ferbue/Image-Adaptive-3DLUT/LUTs/unpaired/exp_'+EXP_ID+'/'
luts_path = base_path + 'saved_models/LUTs_best_fid.pth'
eval_path = base_path + 'evaluation/best_fid_png/'
graphics_folder = base_path + 'graphics/'
if not os.path.isdir(graphics_folder):
    os.mkdir(graphics_folder)


weights_file = eval_path + '/wsi_global_weights.csv'
weights_df = pd.read_csv(weights_file)





# In[4]:

print('Plotting: weights')

wsi_weigts = np.zeros(3)
for i, w in enumerate(['w0','w1','w2']):
    w_hist = plt.hist(weights_df[w], histtype='step')
    wsi_weigts[i]= np.median(weights_df[w])
    plt.vlines(wsi_weigts[i], 0, np.max(w_hist[0]), color= plt.cm.tab10(i), label=w+" = {:.2f}".format(wsi_weigts[i]))
plt.title('Weight distribution and median per LUT')
plt.legend()
plt.savefig(graphics_folder+'Weight_distribution_all.png')


# In[3]:

print('Plotting: weights by scanner')

plt.figure(figsize=(12,5))

source_files = test_df[test_df['scanner_model_new'] == SOURCE_DOMAIN]['file_name'].unique()

filtered_weight_df = weights_df[
    weights_df['file_name'].isin(source_files)
]

plt.subplot(121)
wsi_weigts_source2target = np.zeros(3)
for i, w in enumerate(['w0','w1','w2']):
    w_hist = plt.hist(filtered_weight_df[w], histtype='step')
    wsi_weigts_source2target[i]= np.median(filtered_weight_df[w])
    plt.vlines(wsi_weigts_source2target[i], 0, np.max(w_hist[0]), color= plt.cm.tab10(i), label=w+" = {:.2f}".format(wsi_weigts_source2target[i]))
plt.title('Weight distribution: '+ SOURCE_DOMAIN+' -> ' + TARGET_DOMAIN)
plt.legend()


source_files = test_df[test_df['scanner_model_new'] == TARGET_DOMAIN]['file_name'].unique()

filtered_weight_df = weights_df[
    weights_df['file_name'].isin(source_files)
]

plt.subplot(122)
wsi_weigts_target2target = np.zeros(3)
for i, w in enumerate(['w0','w1','w2']):
    w_hist = plt.hist(filtered_weight_df[w], histtype='step')
    wsi_weigts_target2target[i]= np.median(filtered_weight_df[w])
    plt.vlines(wsi_weigts_target2target[i], 0, np.max(w_hist[0]), color= plt.cm.tab10(i), label=w+" = {:.2f}".format(wsi_weigts_target2target[i]))
plt.title('Weight distribution: '+ SOURCE_DOMAIN)
plt.title('Weight distribution: '+ TARGET_DOMAIN+' -> ' + TARGET_DOMAIN)
plt.legend()

plt.savefig(graphics_folder+'Weight_distribution_by_scanner.png')


# In[7]:


del test_df, filtered_weight_df, weights_df


# In[4]:

print('Plotting: LUTs overview')


lut = torch.load(luts_path, map_location='cpu')
lut_dim = 33
lut0, lut1, lut2 = [lut[str(i)]['LUT'] for i in range(3)]

lut_identity = Generator3DLUT_identity()
lut_base = lut_identity.state_dict()['LUT']
lut_base = lut_base.permute(1, 2, 3, 0)

# convert [3, 17, 17, 17] to [17, 17, 17, 3]
lut0 = lut0.permute(1, 2, 3, 0)
lut1 = lut1.permute(1, 2, 3, 0)
lut2 = lut2.permute(1, 2, 3, 0)

weighted_lut_source2target = wsi_weigts_source2target[0]*lut0+wsi_weigts_source2target[1]*lut1+wsi_weigts_source2target[2]*lut2
clipped_lut = weighted_lut_source2target.detach().clone()
clipped_lut = torch.clamp(clipped_lut, 0, 1)

weighted_lut_target2target = wsi_weigts_target2target[0]*lut0+wsi_weigts_target2target[1]*lut1+wsi_weigts_target2target[2]*lut2
clipped_lut_target2target = weighted_lut_target2target.detach().clone()
clipped_lut_target2target = torch.clamp(clipped_lut_target2target, 0, 1)



# TODO: better ways for this process
# normalization
lut0 = (lut0 - lut0.min()) / (lut0.max() - lut0.min())
lut1 = (lut1 - lut1.min()) / (lut1.max() - lut1.min())
lut2 = (lut2 - lut2.min()) / (lut2.max() - lut2.min())

# weighted_lut = (weighted_lut - weighted_lut.min()) / (weighted_lut.max() - weighted_lut.min())

# visualize the LUT, take lut0 as an example
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(231, projection='3d')
vis_lut_ax(lut0, lut_dim, ax)
plt.title("Normalized LUT 0. w= {:.2f}".format(wsi_weigts[0]))
ax = fig.add_subplot(232, projection='3d')
vis_lut_ax(lut1, lut_dim, ax)
plt.title("Normalized LUT 1. w= {:.2f}".format(wsi_weigts[1]))
ax = fig.add_subplot(233, projection='3d')
vis_lut_ax(lut2, lut_dim, ax)
plt.title("Normalized LUT 2. w= {:.2f}".format(wsi_weigts[2]))

ax = fig.add_subplot(234, projection='3d')
vis_lut_ax(lut_base, lut_dim, ax)
plt.title('Identity LUT')

ax = fig.add_subplot(235, projection='3d')
vis_lut_ax(clipped_lut_target2target, lut_dim, ax)
plt.title('LUT '+TARGET_DOMAIN+' -> '+ TARGET_DOMAIN)

ax = fig.add_subplot(236, projection='3d')
vis_lut_ax(clipped_lut, lut_dim, ax)
plt.title('LUT '+SOURCE_DOMAIN+' -> '+ TARGET_DOMAIN)

plt.savefig(graphics_folder+'All_LUT_overview.png')


# In[ ]:


# plt.figure(figsize=(30,2))
# values = np.linspace(0,1,255)
# for i,p in enumerate(values):
#     plt.subplot(1,len(values),i+1)
#     plt.imshow(p*np.ones([1,1,3])), plt.axis('off')


# In[87]:

print('Plotting: Grey curves')


plt.figure(figsize=(15,8))
titles=['normalized lut0','normalized lut1','normalized lut2','LUT '+SOURCE_DOMAIN+' -> XR','LUT '+TARGET_DOMAIN+' -> XR']
plt.suptitle('RGB curves on grey levels (Diagonal R=G=B)', fontsize=15)
for i, lut2explore in enumerate([lut0, lut1, lut2, clipped_lut, clipped_lut_target2target]):
    N = lut2explore.shape[0]

    curve_r = []
    curve_g = []
    curve_b = []

    idx = torch.arange(lut2explore.shape[0])
    diag = lut2explore[idx, idx, idx]  # shape: [33, 3]

    curve_r = diag[:, 0]
    curve_g = diag[:, 1]
    curve_b = diag[:, 2]
    plt.subplot(2,3,i+1)
    
    plt.plot(np.linspace(0,1,33),lut_base[idx, idx, idx][:,0], color='gray',ls='dashed',alpha=0.5)
    plt.plot(np.linspace(0,1,33),curve_r, color='red')
    plt.plot(np.linspace(0,1,33),curve_g, color='green')
    plt.plot(np.linspace(0,1,33),curve_b, color='blue')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.title(titles[i], fontsize=15)
    plt.savefig(graphics_folder+'Gray_RGB_curves.png')


# In[12]:


# valsH = np.linspace(0, 360, 13)
# valsS = np.linspace(0, 1, 12)

# # Create 3D grid
# H, S = np.meshgrid(valsH, valsS, indexing='ij')

print('Plotting: Polar HSV plot')


def polar_lut_plot(H, S, rgb_list_uint, ax, V=None, title="A line plot on a polar axis"):

    if V is None:
        V = 20*np.ones(len(rgb_list_uint))

    ax.scatter(np.radians(H[:-1,:]), S[:-1,:], c=rgb_list_uint/255, s=V)

    ax.set_rmax(1)
    ax.grid(False)
    ax.set_title(title, va='bottom')

    # Convert to radians once
    theta = np.radians(H)
    r = S

    nH, nS = H.shape

    #Grid of neighbours
    for i in range(nH):
        for j in range(nS):
            
            # --- Neighbor in Hue (wrap around) ---
            i_next = (i + 1) % nH
            ax.plot(
                [theta[i, j], theta[i_next, j]],
                [r[i, j], r[i_next, j]],
                color='gray', alpha=0.5, linewidth=0.8
            )
            
            # --- Neighbor in Saturation (no wrap) ---
            if j < nS - 1:
                ax.plot(
                    [theta[i, j], theta[i, j+1]],
                    [r[i, j], r[i, j+1]],
                    color='gray', alpha=0.5, linewidth=0.8
                )

# fig, axs = plt.subplots(1, 1, figsize=(5, 8), subplot_kw={'projection': 'polar'}, layout='constrained')
# ax = axs
# polar_lut_plot(H, S, rgb_list, ax, V=20*0.8*np.ones(len(rgb_list)), title="Identity grid")


# In[89]:


valsH = np.linspace(0, 360, 13)
valsS = np.linspace(0, 1, 12)

# Create 3D grid
H, S = np.meshgrid(valsH, valsS, indexing='ij')
hsv_list = np.array([H[:-1,:].flatten()*(255/360),
S[:-1,:].flatten()*255,
200*np.ones(len(S[:-1,:].flatten()))],dtype='uint8')
print(hsv_list.shape)
# print(hsv_list)

rgb_list = np.array(PIL.Image.fromarray(hsv_list.reshape(3,1,-1).T,mode='HSV').convert('RGB')).reshape(-1,3)
rgb_index = np.round(rgb_list*(33/255)).astype('int')

rgb_list_lut = np.zeros(rgb_list.shape)
rgb_list_lut_target2target = np.zeros(rgb_list.shape)
rgb_index = np.round(rgb_list*(33/255)).astype('int')
for i in range(len(rgb_list)):
    rgb_list_lut[i,:] = clipped_lut[rgb_index[i,2],rgb_index[i,1],rgb_index[i,0],:]
    rgb_list_lut_target2target[i,:] = clipped_lut_target2target[rgb_index[i,2],rgb_index[i,1],rgb_index[i,0],:]
rgb_list_lut = (rgb_list_lut*255).astype('uint8')
rgb_list_lut_target2target = (rgb_list_lut_target2target*255).astype('uint8')

# hsv_list_lut = np.array(PIL.Image.fromarray(rgb_list_lut.reshape(3,1,-1).T,mode='RGB').convert('HSV')).reshape(-1,3)
hsv_list_lut = rgb2hsv(rgb_list_lut) #all between 0-1
hsv_list_lut_target2target = rgb2hsv(rgb_list_lut_target2target) #all between 0-1


# In[99]:


n_cols = 12
n_rows = 12
n_show = n_cols * n_rows

fig, axes = plt.subplots(n_rows, n_cols*3+2, figsize=(15, 5))

for i in range(n_show):
    r = i // n_cols
    c = i % n_cols

    # Left grid
    axes[r, c].imshow(rgb_list[i].reshape(1,1,3))
    axes[r, c].axis('off')

    # center grid
    axes[r, c + n_cols+1].imshow(rgb_list_lut[i].reshape(1,1,3))
    axes[r, c + n_cols+1].axis('off')

    # right grid
    axes[r, c + 2*n_cols+2].imshow(rgb_list_lut_target2target[i].reshape(1,1,3))
    axes[r, c + 2*n_cols+2].axis('off')

# ---- Titles ----
fig.text(0.17, 1.05, "Original RGB", ha='center', va='center', fontsize=16)
fig.text(0.50, 1.05, "LUT "+SOURCE_DOMAIN +' -> '+ TARGET_DOMAIN , ha='center', va='center', fontsize=16)
fig.text(0.83, 1.05, "LUT "+TARGET_DOMAIN +' -> '+ TARGET_DOMAIN, ha='center', va='center', fontsize=16)

#turn of empty columns as separators
for r in range(n_rows):
    axes[r, n_cols].axis('off')
    axes[r, 2*n_cols+1].axis('off')
# ---- Vertical separator ----
# line = Line2D([0.5, 0.5], [0.05, 0.95], transform=fig.transFigure,
#               color='black', linewidth=2)
# fig.add_artist(line)

plt.tight_layout()
# plt.tight_layout(rect=[0, 0, 1, 0.93])  # leave space for titles
plt.savefig(graphics_folder+'HSV_sampled_palette_original_vs_LUT.png', bbox_inches='tight')


# In[98]:


#SCIKIT
Hlut = np.zeros(H.shape)
Hlut[:-1,:] = hsv_list_lut[:,0].reshape(12,12)*(360)
Hlut[-1,:]=Hlut[0,:]

Slut = np.zeros(S.shape)
Slut[:-1,:] = hsv_list_lut[:,1].reshape(12,12)
Slut[-1,:] = Slut[0,:] 

Vlut = hsv_list_lut[:,2]

# TARGET 2 TARGET
Hlut_target2target = np.zeros(H.shape)
Hlut_target2target[:-1,:] = hsv_list_lut_target2target[:,0].reshape(12,12)*(360)
Hlut_target2target[-1,:]=Hlut_target2target[0,:]

Slut_target2target = np.zeros(S.shape)
Slut_target2target[:-1,:] = hsv_list_lut_target2target[:,1].reshape(12,12)
Slut_target2target[-1,:] = Slut_target2target[0,:] 

Vlut_target2target = hsv_list_lut_target2target[:,2]


# In[100]:


fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': 'polar'},
                        layout='constrained')
ax = axs[0]

default_size = 30
polar_lut_plot(H, S, rgb_list, ax, V=default_size*0.8*np.ones(len(rgb_list)), title="Original grid")
ax = axs[1]
polar_lut_plot(Hlut, Slut, rgb_list_lut, ax, V=default_size*Vlut, title="LUT "+SOURCE_DOMAIN +' -> '+ TARGET_DOMAIN  )
ax = axs[2]
polar_lut_plot(Hlut_target2target, Slut_target2target, rgb_list_lut_target2target, ax, V=default_size*Vlut_target2target, title="LUT "+TARGET_DOMAIN +' -> '+ TARGET_DOMAIN )

plt.savefig(graphics_folder+'HSV_polar_grid_original_vs_LUT.png', bbox_inches='tight')


# In[101]:

print('Plotting: Hue, saturation and value specific differences')


i=11
items=12
x_axis = np.linspace(0,1,items)
plt.figure(figsize=(18,8))
for i in range(12):
    plt.subplot(3,4,i+1)
    offset = np.sin(np.radians(H[i,0]))
    plt.plot(x_axis,np.sin(np.radians(H[i,:]))-offset, c=rgb_list[i*items+items-1]/255,zorder=0)
    plt.scatter(x_axis,np.sin(np.radians(H[i,:]))-offset, c=rgb_list[i*items:i*items+items]/255,zorder=2)

    # plt.plot(x_axis,np.sin(np.radians(Hlut[i,:]))-offset, c=rgb_list_lut[i*items+items-1]/255,zorder=1)
    # plt.scatter(x_axis,np.sin(np.radians(Hlut[i,:]))-offset, c=rgb_list_lut[i*items:i*items+items]/255,zorder=3)
    plt.plot(x_axis,np.sin(np.radians(Hlut[i,:])-np.radians(H[i,:])), c=rgb_list_lut[i*items+items-1]/255,zorder=1)
    plt.scatter(x_axis,np.sin(np.radians(Hlut[i,:])-np.radians(H[i,:])), c=rgb_list_lut[i*items:i*items+items]/255,zorder=3)

    plt.plot(x_axis,np.sin(np.radians(Hlut_target2target[i,:])-np.radians(H[i,:])), c='gray',zorder=1)
    plt.scatter(x_axis,np.sin(np.radians(Hlut_target2target[i,:])-np.radians(H[i,:])), c=rgb_list_lut_target2target[i*items:i*items+items]/255,zorder=3)

    plt.ylim(-1.1,1.1)
    plt.ylabel('Hue difference sin(H-H0)')

    if i % 4 != 0:
        plt.yticks([])
        plt.ylabel("")

    if i>7:
        plt.xlabel("Original saturation level")

plt.tight_layout()
plt.suptitle('Changes in hue. LUT saturation is ignored.', y=1.05)
plt.savefig(graphics_folder+'Hue_changes.png', bbox_inches='tight')


# In[103]:


i=11
items=12
x_axis = np.linspace(0,1,items)
plt.figure(figsize=(18,8))
for i in range(12):
    plt.subplot(3,4,i+1)
    offset = np.sin(np.radians(H[i,0]))
    # plt.plot(range(items),np.sin(np.radians(H[i,:]))-offset, c=rgb_list[i*items+items-1]/255,zorder=0)
    plt.plot(x_axis,S[i,:], c=rgb_list[i*items+items-1]/255,zorder=0)
    plt.scatter(x_axis,S[i,:], c=rgb_list[i*items:i*items+items]/255,zorder=2)

    plt.plot(x_axis,Slut_target2target[i,:], c='gray',zorder=0)
    plt.scatter(x_axis,Slut_target2target[i,:], c=rgb_list_lut_target2target[i*items:i*items+items]/255,zorder=2)

    plt.plot(x_axis,Slut[i,:], c=rgb_list_lut[i*items+items-1]/255,zorder=0)
    plt.scatter(x_axis,Slut[i,:], c=rgb_list_lut[i*items:i*items+items]/255,zorder=2)
    # plt.ylim(-1,1)



    plt.ylabel('Saturation')
    # plt.axis('equal')

    if i % 4 != 0:
        plt.yticks([])
        plt.ylabel("")

    if i>7:
        plt.xlabel("Original saturation level")

plt.tight_layout()
plt.suptitle('Changes in Saturation. LUT Hue is ignored.', y=1.05)
plt.savefig(graphics_folder+'Saturation_changes.png', bbox_inches='tight')


# In[104]:


i=11
items=12
x_axis = np.linspace(0,1,items)
plt.figure(figsize=(20,8))
for i in range(12):
    plt.subplot(3,4,i+1)
    offset = np.sin(np.radians(H[i,0]))
    # plt.plot(range(items),np.sin(np.radians(H[i,:]))-offset, c=rgb_list[i*items+items-1]/255,zorder=0)
    plt.plot(x_axis,0.8*np.ones(items), c=rgb_list[i*items+items-1]/255,zorder=0)
    plt.scatter(x_axis,0.8*np.ones(items), c=rgb_list[i*items:i*items+items]/255,zorder=2)

    plt.plot(x_axis,Vlut_target2target[i*items:i*items+items], c='gray',zorder=0)
    plt.scatter(x_axis,Vlut_target2target[i*items:i*items+items], c=rgb_list_lut_target2target[i*items:i*items+items]/255,zorder=2)
    
    plt.plot(x_axis,Vlut[i*items:i*items+items], c=rgb_list_lut[i*items+items-1]/255,zorder=0)
    plt.scatter(x_axis,Vlut[i*items:i*items+items], c=rgb_list_lut[i*items:i*items+items]/255,zorder=2)
    plt.ylim(0.45,1)

    plt.ylabel('Value')
    if i % 4 != 0:
        plt.yticks([])
        plt.ylabel("")

    if i>7:
        plt.xlabel("Original saturation level")
    
plt.tight_layout()
plt.suptitle('Changes in Value. LUT Hue, Sat are ignored.', y=1.05)
plt.savefig(graphics_folder+'Value_changes.png', bbox_inches='tight')


# In[105]:
print('Plotting: Full LUT polar plot')


lut_base.shape #torch.Size([33, 33, 33, 3])
lut_base_flat = lut_base.reshape(-1,3) #input_color (0, 1)

lut_base_pil = lut_base.reshape(-1,1,3).T
print(lut_base_pil.shape)
lut_base_pil = torchvision.transforms.functional.to_pil_image(lut_base_pil)
lut_base_hsv = lut_base_pil.convert('HSV')

lut_base_hsv = np.array(lut_base_hsv)
lut_base_hsv = lut_base_hsv.squeeze()

clipped_lut_flat = clipped_lut.reshape(-1,3) #input_color (0, 1)
clipped_lut_pil = clipped_lut.reshape(-1,1,3).T

clipped_lut_pil = torchvision.transforms.functional.to_pil_image(clipped_lut_pil)
clipped_lut_hsv = clipped_lut_pil.convert('HSV')

clipped_lut_hsv = np.array(clipped_lut_hsv)
clipped_lut_hsv = clipped_lut_hsv.squeeze()

clipped_lut_flat_target2target = clipped_lut_target2target.reshape(-1,3) #input_color (0, 1)
clipped_lut_pil_target2target = clipped_lut_target2target.reshape(-1,1,3).T

clipped_lut_pil_target2target = torchvision.transforms.functional.to_pil_image(clipped_lut_pil_target2target)
clipped_lut_hsv_target2target = clipped_lut_pil_target2target.convert('HSV')

clipped_lut_hsv_target2target = np.array(clipped_lut_hsv_target2target)
clipped_lut_hsv_target2target = clipped_lut_hsv_target2target.squeeze()


# In[107]:



fig, axs = plt.subplots(1, 3,  figsize=(15, 8), subplot_kw={'projection': 'polar'},
                        layout='constrained')
ax = axs[0]
ax.scatter(np.radians(lut_base_hsv[:,0]*(360/255)),lut_base_hsv[:,1]/255, c = lut_base_flat, s=2, alpha=0.5)
# ax.set_rmax(1)
ax.set_rticks([])
ax.set_xticks([])
ax.grid(True)
ax.set_title("Unitary LUT", va='bottom')

ax = axs[1]
ax.scatter(np.radians(clipped_lut_hsv[:,0]*(360/255)),clipped_lut_hsv[:,1]/255, c = clipped_lut_flat, s=2, alpha=0.5)
# ax.set_rmax(1)
ax.set_rticks([])
ax.set_xticks([])
ax.grid(True)
ax.set_title("LUT "+ SOURCE_DOMAIN + " ->  "+ TARGET_DOMAIN, va='bottom')

ax = axs[2]
ax.scatter(np.radians(clipped_lut_hsv_target2target[:,0]*(360/255)),clipped_lut_hsv_target2target[:,1]/255, c = clipped_lut_flat_target2target, s=2, alpha=0.5)
# ax.set_rmax(1)
ax.set_rticks([])
ax.set_xticks([])
ax.grid(True)
ax.set_title("LUT "+ TARGET_DOMAIN + " ->  "+ TARGET_DOMAIN, va='bottom')

plt.savefig(graphics_folder+'HSV_polar_full_LUT.png', bbox_inches='tight')


# In[ ]:


# fig, axs = plt.subplots(1, 2,  figsize=(10, 8), subplot_kw={'projection': 'polar'},
#                         layout='constrained')
# ax = axs[0]
# ax.scatter(np.radians(lut_base_hsv[:,0]*(360/255)),lut_base_hsv[:,1]/255, c = lut_base_flat, s=2, alpha=0.5)
# # ax.set_rmax(1)
# ax.set_rticks([])
# ax.set_xticks([])
# ax.grid(True)
# ax.set_title("Unitary LUT", va='bottom')

# ax = axs[1]
# ax.scatter(np.radians(clipped_lut_hsv[:,0]*(360/255)),clipped_lut_hsv[:,1]/255, c = clipped_lut_flat, s=2, alpha=0.5)
# # ax.set_rmax(1)
# ax.set_rticks([])
# ax.set_xticks([])
# ax.grid(True)
# ax.set_title("Estimated LUT Philips -> XR", va='bottom')

# plt.savefig(graphics_folder+'Value_changes.png')


# In[13]:

print('Plotting: Gifs')
#Gifs
import matplotlib.animation as animation


# In[ ]:


names = ['lut0','lut1','lut2','weighted_lut']
for l, lut_to_explore in enumerate([lut0, lut1, lut2, clipped_lut]):
    print(names[l])
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')


    def animate(i):
        ax.cla()
        vis_lut_ax(lut_to_explore, lut_dim, ax, b_stop=i)
        return ax

    ani = animation.FuncAnimation(fig, animate, frames=lut_dim, interval=200)

    # plt.show()

    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=7,
                                    metadata=dict(artist='FernandoPB'),
                                    bitrate=1800)
    ani.save(graphics_folder+ names[l] + '.gif', writer=writer) # 1.29 min


# In[5]:


lut_base_flat = lut_base.reshape(-1,3)
clipped_lut_flat = clipped_lut.reshape(-1,3)
clipped_lut_flat_target2target = clipped_lut_target2target.reshape(-1,3)


# In[ ]:


# step=2

# fig = plt.figure(figsize=(25,10))

# ax = fig.add_subplot(131, projection='3d') 
# ax.scatter(lut_base_flat[::step,2],lut_base_flat[::step,1],lut_base_flat[::step,0], c= lut_base_flat[::step,:], s=2, alpha=1)
# ax.set_title('Unitary RGB space')

# ax2 = fig.add_subplot(132, projection='3d') 
# ax2.scatter(clipped_lut_flat[::step,2],clipped_lut_flat[::step,1],clipped_lut_flat[::step,0], c= clipped_lut_flat[::step,:], s=2, alpha=1)
# ax2.set_title("LUT "+ SOURCE_DOMAIN + " ->  "+ TARGET_DOMAIN)

# ax3 = fig.add_subplot(133, projection='3d') 
# ax3.scatter(clipped_lut_flat_target2target[::step,2],clipped_lut_flat_target2target[::step,1],clipped_lut_flat_target2target[::step,0], c= clipped_lut_flat_target2target[::step,:], s=2, alpha=1)
# ax3.set_title("LUT "+ TARGET_DOMAIN + " ->  "+ TARGET_DOMAIN)

# for axis in [ax, ax2, ax3]:
# # 
#     axis.set_xlabel('B')
#     axis.set_xlim(0,1)
#     axis.set_ylabel('G')
#     axis.set_ylim(0,1)
#     axis.set_zlabel('R')
#     axis.set_zlim(0,1)


# In[14]:

print('Plotting: Rotating gif')
step=1

fig = plt.figure(figsize=(25,10))

ax = fig.add_subplot(131, projection='3d') 
ax.scatter(lut_base_flat[::step,2],lut_base_flat[::step,1],lut_base_flat[::step,0], c= lut_base_flat[::step,:], s=2, alpha=1)
ax.set_title('Unitary RGB space')

ax2 = fig.add_subplot(132, projection='3d') 
ax2.scatter(clipped_lut_flat[::step,2],clipped_lut_flat[::step,1],clipped_lut_flat[::step,0], c= clipped_lut_flat[::step,:], s=2, alpha=1)
ax2.set_title("LUT "+ SOURCE_DOMAIN + " ->  "+ TARGET_DOMAIN)

ax3 = fig.add_subplot(133, projection='3d') 
ax3.scatter(clipped_lut_flat_target2target[::step,2],clipped_lut_flat_target2target[::step,1],clipped_lut_flat_target2target[::step,0], c= clipped_lut_flat_target2target[::step,:], s=2, alpha=1)
ax3.set_title("LUT "+ TARGET_DOMAIN + " ->  "+ TARGET_DOMAIN)

for axis in [ax, ax2, ax3]:

    axis.set_xlabel('B')
    axis.set_xlim(0,1)
    axis.set_ylabel('G')
    axis.set_ylim(0,1)
    axis.set_zlabel('R')
    axis.set_zlim(0,1)
def rotate(angle):
    ax.view_init(elev=30, azim=angle)
    ax2.view_init(elev=30, azim=angle)
    ax3.view_init(elev=30, azim=angle)


ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0,360,2), interval=100)

ani.save(graphics_folder+"lut_rotation_all.gif", writer="pillow")
# ani.save(graphics_folder+"lut_rotation.mp4", writer="ffmpeg", dpi=200)


# In[ ]:
print('Plotting: Progressive gif')

base = lut_base_flat
dest = clipped_lut_flat

mag = np.linalg.norm(dest-base, axis=1)
# mag = mag/np.max(mag)

mag_sort = np.argsort(mag)[::-1].copy()

N = len(base)
n_samples = 100
n_frames = int(np.ceil(N / n_samples))

fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(1,2,1, projection='3d')
ax2 = fig.add_subplot(1,2,2, projection='3d')


def setup_ax(ax):
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)
    ax.set_box_aspect([1,1,1])

    angle=45
    ax.view_init(elev=30, azim=angle)


def update(frame):
    frame = frame-3
    end = 0
    
    ax1.cla()
    ax2.cla()
    ax3.cla()

    setup_ax(ax1)
    setup_ax(ax2)
    setup_ax(ax3)

    if frame>=0:
        end = min((frame+1)*n_samples, N)
        idxs = mag_sort[:end]

        filtered_base = base[idxs]
        filtered_dest = dest[idxs]

        ax1.scatter(
            filtered_base[:,2],
            filtered_base[:,1],
            filtered_base[:,0],
            c=filtered_base,
            s=5
        )

        ax2.scatter(
            filtered_dest[:,2],
            filtered_dest[:,1],
            filtered_dest[:,0],
            c=filtered_dest,
            s=5
        )


    ax1.set_title(f"Identity")
    ax2.set_title(f"LUT "+SOURCE_DOMAIN+ ' ->' + TARGET_DOMAIN)

    fig.suptitle("{:.3f} % \n points: {} / {}".format(100*end/N, end, N))

# update(100)

ani = animation.FuncAnimation(fig, update, frames=n_frames+3, interval=200)

ani.save(graphics_folder+"lut_progressive_points2.gif", writer=animation.PillowWriter(fps=5))


# In[ ]:
print('Plotting: Progressive gif 2')

base = lut_base_flat
dest = clipped_lut_flat_target2target

mag = np.linalg.norm(dest-base, axis=1)
# mag = mag/np.max(mag)

mag_sort = np.argsort(mag)[::-1].copy()

N = len(base)
n_samples = 100
n_frames = int(np.ceil(N / n_samples))

fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(1,2,1, projection='3d')
ax2 = fig.add_subplot(1,2,2, projection='3d')


def setup_ax(ax):
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)
    ax.set_box_aspect([1,1,1])

    angle=45
    ax.view_init(elev=30, azim=angle)


def update(frame):
    frame = frame-3
    end = 0

    ax1.cla()
    ax2.cla()
    ax3.cla()

    setup_ax(ax1)
    setup_ax(ax2)
    setup_ax(ax3)

    if frame>=0:

        end = min((frame+1)*n_samples, N)
        idxs = mag_sort[:end]

        filtered_base = base[idxs]
        filtered_dest = dest[idxs]

        ax1.scatter(
            filtered_base[:,2],
            filtered_base[:,1],
            filtered_base[:,0],
            c=filtered_base,
            s=5
        )

        ax2.scatter(
            filtered_dest[:,2],
            filtered_dest[:,1],
            filtered_dest[:,0],
            c=filtered_dest,
            s=5
        )


    ax1.set_title(f"Identity")
    ax2.set_title(f"LUT "+TARGET_DOMAIN+ ' ->' + TARGET_DOMAIN)

    fig.suptitle("{:.3f} % \n points: {} / {}".format(100*end/N, end, N))

# update(3)

ani = animation.FuncAnimation(fig, update, frames=n_frames+3, interval=200)

ani.save(graphics_folder+"lut_progressive_points_target.gif", writer=animation.PillowWriter(fps=5))

print('COMPLETED')
# In[ ]:


# mag = np.linalg.norm(clipped_lut_flat-lut_base_flat, axis=1)
# mag = mag/np.max(mag)

# mag_sort = np.argsort(mag)[::-1].copy()

# n_samples = 50
# fig = plt.figure(figsize=(10,5))

# i=0
# idxs = mag_sort[i*n_samples:(i+1)*n_samples]
# filtered_base = lut_base_flat[idxs,:]
# filtered_clipped_lut = clipped_lut_flat[idxs,:]
# filtered_mag = mag[idxs]

# # print(len(filtered_base))

# ax1 = fig.add_subplot(1,2, i+1, projection='3d')
# ax1.scatter(filtered_base[:,2],filtered_base[:,1],filtered_base[:,0], c= filtered_base, alpha=1, s=5) #, alpha=filtered_mag
# plt.title('RGB: ' + str(i*n_samples)+'->'+str((i+1)*n_samples))

# ax2 = fig.add_subplot(1,2, i+2, projection='3d')
# ax2.scatter(filtered_clipped_lut[:,2],filtered_clipped_lut[:,1],filtered_clipped_lut[:,0], c= filtered_clipped_lut, alpha=1, s=5) #, alpha=filtered_mag

# # ax.scatter(lut_base_flat[::step,2],lut_base_flat[::step,1],lut_base_flat[::step,0], c= lut_base_flat[::step,:], s=2, alpha=mag[::step])
# plt.title('LUT ' +str(i*n_samples)+'->'+str((i+1)*n_samples))

# for ax in [ax1,ax2]:
#     ax.set_xlabel("R")
#     ax.set_ylabel("G")
#     ax.set_zlabel("B")

#     ax.set_xlim(0,1)
#     ax.set_ylim(0,1)
#     ax.set_zlim(0,1)
# plt.tight_layout()


# In[ ]:


# # --- convert tensors if needed ---
# base = lut_base_flat.detach().cpu().numpy() if hasattr(lut_base_flat, "detach") else lut_base_flat
# dest = clipped_lut_flat.detach().cpu().numpy() if hasattr(clipped_lut_flat, "detach") else clipped_lut_flat

# vec = dest - base

# # --- subsample to avoid clutter ---
# step = 10
# base = base[::step]
# vec = vec[::step]
# dest = dest[::step]

# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111, projection='3d')


# mag = np.linalg.norm(vec, axis=1)


# ax.scatter(dest[:,0], dest[:,1], dest[:,2], c=plt.cm.inferno(mag / mag.max()), s=5,zorder=2)

# ax.set_xlabel("R")
# ax.set_ylabel("G")
# ax.set_zlabel("B")

# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
# ax.set_zlim(0,1)

# ax.set_box_aspect([1,1,1])

# plt.show()


# In[ ]:


# from scipy.spatial import ConvexHull

# lut2explore = clipped_lut_flat.detach().numpy()
# hull = ConvexHull(lut2explore)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# for simplex in hull.simplices:
#     ax.plot(lut2explore[simplex,0],
#             lut2explore[simplex,1],
#             lut2explore[simplex,2],
#             'k-')

# # plt.show()


# In[ ]:


# from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# points = clipped_lut_flat.detach().numpy()   # your Nx3 array

# # hull = ConvexHull(points)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Collect triangular faces
# faces = [points[simplex] for simplex in hull.simplices]

# poly = Poly3DCollection(
#     faces,
#     facecolor='lightgray',
#     edgecolor='k',
#     alpha=0.8
# )

# ax.add_collection3d(poly)

# # ax.scatter(points[:,0], points[:,1], points[:,2], s=2)

# ax.set_xlim(points[:,0].min(), points[:,0].max())
# ax.set_ylim(points[:,1].min(), points[:,1].max())
# ax.set_zlim(points[:,2].min(), points[:,2].max())

# plt.show()


# In[ ]:


# from mpl_toolkits.mplot3d import Axes3D
# # import matplotlib.pyplot as plt
# import matplotlib.tri as mtri
# # import numpy as np


# In[ ]:


# lut2explore = clipped_lut_flat.detach().numpy()

# x = lut2explore[:,0]
# y = lut2explore[:,1]
# z = lut2explore[:,2]

# tri = mtri.Triangulation(x, y)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot_trisurf(tri, z, cmap='viridis', alpha=0.8)

# plt.show()

