#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap.umap_ as umap
import pickle
import torch
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
import glob
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import argparse


##### Main #######
parser = argparse.ArgumentParser(description='Feature comparison for LUT transformed images.')

parser.add_argument('--exp_id',  dest='exp_id',  type=str, default='exp_id', help='id for this experiment')
parser.add_argument('--source_domain',  dest='source_domain',  type=str, default='PHILIPS', help='Scanner we transform from')
parser.add_argument('--target_domain',  dest='target_domain',  type=str, default='XR', help='Scanner we transform to')
parser.add_argument('--feat_csv',  dest='feat_csv',  type=str, default='scanb_malmo_source_exp20_bothscans_uni.pkl', help='pkl file with UNI features for the data')
parser.add_argument('--feat_lut_csv',  dest='feat_lut_csv',  type=str, default='scanb_malmo_lut_exp20_bothscans_uni.pkl', help='pkl file with LUT transformed UNI features')
parser.add_argument('--feat_macenko_csv',  dest='feat_macenko_csv',  type=str, default='scanb_malmo_macenko_bothscans_uni.pkl', help='pkl file with MACENKO transformed UNI features')
parser.add_argument('--test_df_path',  dest='test_df_path',  type=str, default='/mnt/ssd/ferbue/Image-Adaptive-3DLUT/dataframes/lut_exp20_bothscanners_v2.csv', help='Full path to csv file with data info (mostly scanner)')

args               = parser.parse_args()

EXP_ID = args.exp_id
SOURCE_DOMAIN = args.source_domain
TARGET_DOMAIN = args.target_domain
base_folder = '/mnt/ssd/ferbue/Image-Adaptive-3DLUT/LUTs/unpaired/exp_'+ EXP_ID +'/'
feature_folder = base_folder + 'uni/scanb_malmo/h224_w224_zdim1024/'
feat_csv = feature_folder + (args.feat_csv).replace(feature_folder,'')
feat_lut_csv = feature_folder + args.feat_lut_csv.replace(feature_folder,'')
feat_macenko_csv = feature_folder + args.feat_macenko_csv.replace(feature_folder,'')
test_df_path = args.test_df_path




def scatter_hist(x, y, ax, ax_histx, ax_histy, alpha=0.5, label='', extras=False):
    #Plot scatter plot + histogram per axis
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, alpha=alpha, label=label, s=1)

    # now determine nice limits by hand:
    binwidth = 0.5
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth
    

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=20, histtype='step')
    ax_histy.hist(y, bins=20, histtype='step', orientation='horizontal')

    if extras and len(x)>1:
        ax.vlines(np.mean(x), np.min(y), np.max(y), ls=':', color='k')
        ax.hlines(np.mean(y), np.min(x), np.max(x), ls=':', color='k', label='mean')

        ax.vlines(np.median(x), np.min(y), np.max(y), ls='dashed', color='darkgrey')
        ax.hlines(np.median(y), np.min(x), np.max(x), ls='dashed', color='darkgrey', label='median')

def scatter_comparison(feat1, feat2, label1, label2, title, extras=False):
    fig, axs = plt.subplot_mosaic([['histx', 'legend'],
                               ['scatter', 'histy']],
                              figsize=(6, 6),
                              width_ratios=(4, 1), height_ratios=(1, 4),
                              layout='constrained')
    scatter_hist(feat1[:,0],feat1[:,1], axs['scatter'], axs['histx'], axs['histy'], label=label1,extras=extras)
    scatter_hist(feat2[:,0],feat2[:,1], axs['scatter'], axs['histx'], axs['histy'], label=label2,extras=extras)

    axs['legend'].axis('off')  # hide axis frame
    axs['legend'].legend(*axs['scatter'].get_legend_handles_labels(),
                        loc='center')

    fig.suptitle(title)

def scatter_comparison_n(feat_list, label_list, title, extras=False):
    fig, axs = plt.subplot_mosaic([['histx', 'legend'],
                               ['scatter', 'histy']],
                              figsize=(6, 6),
                              width_ratios=(4, 1), height_ratios=(1, 4),
                              layout='constrained')
    for i in range(len(feat_list)):
        feat = feat_list[i]
        scatter_hist(feat[:,0],feat[:,1], axs['scatter'], axs['histx'], axs['histy'], label=label_list[i],extras=extras)


    axs['legend'].axis('off')  # hide axis frame
    axs['legend'].legend(*axs['scatter'].get_legend_handles_labels(),
                        loc='center')

    fig.suptitle(title)

def wasserstein_distance_sum(feat1, feat2):
    n_dim = feat1.shape[1]
    w_distances = np.zeros(n_dim)
    for i in range(n_dim):
        w_distances[i]= wasserstein_distance(feat1[:,i], feat2[:,i])
    return w_distances.sum()


# In[ ]:


#Get all features:
usecols = ["tile_name", "scanner_model", 'macenko_blur', 'crude_tile_path','png_tile_path','macenko_tile_path']





graphics_folder = base_folder + 'feat_comparison/'
if not os.path.isdir(graphics_folder):
    os.mkdir(graphics_folder)
    
print('Reading all the stuff')


test_df = pd.read_csv(test_df_path, usecols=usecols) #1742609 rows

features = pd.read_pickle(feat_csv)
features2 = pd.merge(features, test_df, on="tile_name") #to get the scanner_model linked to the features
feats = features2.drop(columns=usecols).to_numpy() #numerical feats only

#list of indexes for each scanner
phil_indexes = (features2['scanner_model']==SOURCE_DOMAIN).to_list()
xr_indexes = [not elem for elem in phil_indexes]
# xr1_idxs = (features2['scanner_model']=='XR1').to_list()
# xr2_idxs = (features2['scanner_model']=='XR2').to_list()

#LUT features,
features_lut = pd.read_pickle(feat_lut_csv)
features_lut_scanner = pd.merge(features_lut, features2[['tile_name','scanner_model']], on="tile_name")
feat_lut= features_lut_scanner.drop(columns=['tile_name','scanner_model']).to_numpy()

#MACENKO features,
features_macenko = pd.read_pickle(feat_macenko_csv)
features_macenko_scanner = pd.merge(features_macenko, features2[['tile_name','scanner_model']], on="tile_name")
feat_macenko= features_macenko_scanner.drop(columns=['tile_name','scanner_model']).to_numpy()


# In[3]:

print('WSI aggregation')
#WSI aggregation
features2['file_name'] = features2['tile_name'].str.split('_clean').str[0]
df_grouped = (
    features2.groupby('file_name')
      .agg(lambda x: x.mean() if x.dtype != 'object' else x.iloc[0])
      .reset_index()
)

wsi_feats = df_grouped.drop(columns=usecols +['file_name']).to_numpy()
print('Source', wsi_feats.shape) # N, 1024

# WSI indexes for each scanner
wsi_philips_idxs = (df_grouped['scanner_model']==SOURCE_DOMAIN).to_list()
wsi_xr_idxs = [not elem for elem in wsi_philips_idxs]
wsi_xr1_idxs = (df_grouped['scanner_model']=='XR1').to_list()
wsi_xr2_idxs = (df_grouped['scanner_model']=='XR2').to_list()


features_lut_scanner['file_name'] = features_lut_scanner['tile_name'].str.split('_clean').str[0]
grouped_lut = (
    features_lut_scanner.groupby('file_name')
      .agg(lambda x: x.mean() if x.dtype != 'object' else x.iloc[0])
      .reset_index()
)
wsi_feat_lut = grouped_lut.drop(columns=['file_name','tile_name','scanner_model']).to_numpy()
print('LUT',wsi_feat_lut.shape)

features_macenko_scanner['file_name'] = features_macenko_scanner['tile_name'].str.split('_clean').str[0]
grouped_macenko = (
    features_macenko_scanner.groupby('file_name')
      .agg(lambda x: x.mean() if x.dtype != 'object' else x.iloc[0])
      .reset_index()
)
wsi_feat_macenko = grouped_macenko.drop(columns=['file_name','tile_name','scanner_model']).to_numpy()
print('Macenko',wsi_feat_macenko.shape)


# In[ ]:

print('Tile sampling')
n_samples_per_wsi=1000
# tile_feat_sampled = features2.groupby("file_name", group_keys=False, sort=False).sample(n=n_samples_per_wsi, random_state=42) #Crashes if not enough tiles available
tile_feat_sampled = (
    features2.groupby("file_name", group_keys=False)
    .apply(lambda x: x.sample(n=min(len(x), n_samples_per_wsi), random_state=42))
)
tile_feat_sampled_lut = features_lut_scanner.loc[tile_feat_sampled.index].drop(columns=['tile_name','scanner_model','file_name']).to_numpy()
tile_feat_sampled_macenko = features_macenko_scanner.loc[tile_feat_sampled.index].drop(columns=['tile_name','scanner_model','file_name']).to_numpy()

phil_indexes_sampled = (tile_feat_sampled['scanner_model']==SOURCE_DOMAIN).to_list()
xr_indexes_sampled = [not elem for elem in phil_indexes_sampled]
tile_feat_sampled = tile_feat_sampled.drop(columns=usecols +['file_name']).to_numpy()


# RAW feature comparison

rows = []
row_identifier_list = []
labels = ['crude_vs_crude',
        'LUT_' + SOURCE_DOMAIN + '_vs_crude_' + TARGET_DOMAIN,
        'LUT_' + SOURCE_DOMAIN +'_vs_' + TARGET_DOMAIN,
        'Macenko_S_vs_T',
        'Source_LUT_vs_Crude',
        'Target_LUT_vs_Crude'
        ]

print('Running RAW Tile')

row_identifier = 'Tile'+'_UNI' 
row_identifier_list.append(row_identifier)

metrics = np.zeros(6)
i=0
value = wasserstein_distance_sum(tile_feat_sampled[phil_indexes_sampled,:], tile_feat_sampled[xr_indexes_sampled,:])
metrics[i]= value
i+=1
# print('crude vs crude: {:.2f}'.format(value)) #9min
value = wasserstein_distance_sum(tile_feat_sampled_lut[phil_indexes_sampled,:], tile_feat_sampled[xr_indexes_sampled,:])
metrics[i]= value
i+=1
# print('LUT ' + SOURCE_DOMAIN + ' vs crude XR: {:.2f}'.format(value))
value = wasserstein_distance_sum(tile_feat_sampled_lut[phil_indexes_sampled,:], tile_feat_sampled_lut[xr_indexes_sampled,:])
metrics[i]= value
i+=1
# print('LUT ' + SOURCE_DOMAIN+ ' vs LUT XR: {:.2f}'.format(value))
value = wasserstein_distance_sum(tile_feat_sampled_macenko[phil_indexes_sampled,:], tile_feat_sampled_macenko[xr_indexes_sampled,:])
metrics[i]= value
i+=1
# print('MACENKO ' + SOURCE_DOMAIN+ ' vs XR: {:.2f}'.format(value))
value = wasserstein_distance_sum(tile_feat_sampled_lut[phil_indexes_sampled,:], tile_feat_sampled[phil_indexes_sampled,:])
metrics[i]= value
i+=1
# print(SOURCE_DOMAIN +' LUT vs crude: {:.2f}'.format(value))
value = wasserstein_distance_sum(tile_feat_sampled_lut[xr_indexes_sampled,:], tile_feat_sampled[xr_indexes_sampled,:])
metrics[i]= value
i+=1
# print('XR LUT vs crude: {:.2f}'.format(value))

print(labels)
print(metrics)
rows.append(metrics)
# [print('{:.2f}'.format(m), end=',') for m in metrics]

for m in metrics: 
    print('{:.2f}'.format(m), end=',')

print('Running RAW WSI')

row_identifier = 'WSI'+'_UNI' 
row_identifier_list.append(row_identifier)


metrics = np.zeros(6)
i=0
value = wasserstein_distance_sum(wsi_feats[wsi_philips_idxs,:], wsi_feats[wsi_xr_idxs,:])
metrics[i]= value
i+=1
# print('crude vs crude: {:.2f}'.format(value)) #9min
value = wasserstein_distance_sum(wsi_feat_lut[wsi_philips_idxs,:], wsi_feats[wsi_xr_idxs,:])
metrics[i]= value
i+=1
# print('LUT ' + SOURCE_DOMAIN + ' vs crude XR: {:.2f}'.format(value))
value = wasserstein_distance_sum(wsi_feat_lut[wsi_philips_idxs,:], wsi_feat_lut[wsi_xr_idxs,:])
metrics[i]= value
i+=1
# print('LUT ' + SOURCE_DOMAIN+ ' vs LUT XR: {:.2f}'.format(value))
value = wasserstein_distance_sum(wsi_feat_macenko[wsi_philips_idxs,:], wsi_feat_macenko[wsi_xr_idxs,:])
metrics[i]= value
i+=1
# print('MACENKO ' + SOURCE_DOMAIN+ ' vs XR: {:.2f}'.format(value))
value = wasserstein_distance_sum(wsi_feat_lut[wsi_philips_idxs,:], wsi_feats[wsi_philips_idxs,:])
metrics[i]= value
i+=1
# print(SOURCE_DOMAIN +' LUT vs crude: {:.2f}'.format(value))
value = wasserstein_distance_sum(wsi_feat_lut[wsi_xr_idxs,:], wsi_feats[wsi_xr_idxs,:])
metrics[i]= value
i+=1
# print('XR LUT vs crude: {:.2f}'.format(value))

print(labels)
print(metrics)
rows.append(metrics)
# [print('{:.2f}'.format(m), end=',') for m in metrics]

for m in metrics: 
    print('{:.2f}'.format(m), end=',')



# In[ ]:

print('Running UMAP ')
UMAP_domain_list = ['Source', 'LUT', 'MACENKO']
level_list = ['Tile', 'WSI']
# rows = []
# row_identifier_list = []
labels = ['crude_vs_crude',
        'LUT_' + SOURCE_DOMAIN + '_vs_crude_' + TARGET_DOMAIN,
        'LUT_' + SOURCE_DOMAIN +'_vs_' + TARGET_DOMAIN,
        'Macenko_S_vs_T',
        'Source_LUT_vs_Crude',
        'Target_LUT_vs_Crude'
        ]
for fg, training_feat_set in enumerate([tile_feat_sampled, tile_feat_sampled_lut, tile_feat_sampled_macenko, wsi_feats, wsi_feat_lut, wsi_feat_macenko]):
    f=fg%3
    level = level_list[fg//3]
    row_identifier = level+'_UMAP_'+UMAP_domain_list[f] 
    row_identifier_list.append(row_identifier)
    print('UMAP DOMAIN: '+UMAP_domain_list[f])
    reducer = umap.UMAP(n_components=10, random_state=42)
    reducer.fit(training_feat_set) #4min with 0.2 

    feats_umap = reducer.transform(tile_feat_sampled) # 12 min
    feats_umap_lut = reducer.transform(tile_feat_sampled_lut) # 20 min both > 38 min
    feats_umap_macenko = reducer.transform(tile_feat_sampled_macenko) # 23 min

    metrics = np.zeros(6)

    i=0
    value = wasserstein_distance_sum(feats_umap[phil_indexes_sampled,:], feats_umap[xr_indexes_sampled,:])
    metrics[i]= value
    i+=1
    # print('crude vs crude: {:.2f}'.format(value)) #9min
    value = wasserstein_distance_sum(feats_umap_lut[phil_indexes_sampled,:], feats_umap[xr_indexes_sampled,:])
    metrics[i]= value
    i+=1
    # print('LUT ' + SOURCE_DOMAIN + ' vs crude XR: {:.2f}'.format(value))
    value = wasserstein_distance_sum(feats_umap_lut[phil_indexes_sampled,:], feats_umap_lut[xr_indexes_sampled,:])
    metrics[i]= value
    i+=1
    # print('LUT ' + SOURCE_DOMAIN+ ' vs LUT XR: {:.2f}'.format(value))
    value = wasserstein_distance_sum(feats_umap_macenko[phil_indexes_sampled,:], feats_umap_macenko[xr_indexes_sampled,:])
    metrics[i]= value
    i+=1
    # print('MACENKO ' + SOURCE_DOMAIN+ ' vs XR: {:.2f}'.format(value))
    value = wasserstein_distance_sum(feats_umap_lut[phil_indexes_sampled,:], feats_umap[phil_indexes_sampled,:])
    metrics[i]= value
    i+=1
    # print(SOURCE_DOMAIN +' LUT vs crude: {:.2f}'.format(value))
    value = wasserstein_distance_sum(feats_umap_lut[xr_indexes_sampled,:], feats_umap[xr_indexes_sampled,:])
    metrics[i]= value
    i+=1
    # print('XR LUT vs crude: {:.2f}'.format(value))

    print(labels)
    print(metrics)
    rows.append(metrics)
    # [print('{:.2f}'.format(m), end=',') for m in metrics]

    for m in metrics: 
        print('{:.2f}'.format(m), end=',')

    i=0
    scatter_comparison(feats_umap[phil_indexes_sampled,:2], feats_umap[xr_indexes_sampled,:2], 'Crude '+SOURCE_DOMAIN, 'Crude '+TARGET_DOMAIN, UMAP_domain_list[f] +' UMAP - ' +level+ ' level')
    plt.savefig(graphics_folder+row_identifier+'_'+labels[i]+'.png')
    i+=1
    scatter_comparison(feats_umap_lut[phil_indexes_sampled,:2], feats_umap[xr_indexes_sampled,:2], 'LUT '+SOURCE_DOMAIN, 'Crude '+TARGET_DOMAIN, UMAP_domain_list[f] +' UMAP - ' +level+ ' level')
    plt.savefig(graphics_folder+row_identifier+'_'+labels[i]+'.png')
    i+=1
    scatter_comparison(feats_umap_lut[phil_indexes_sampled,:2], feats_umap_lut[xr_indexes_sampled,:2], 'LUT '+SOURCE_DOMAIN, 'LUT '+TARGET_DOMAIN, UMAP_domain_list[f] +' UMAP - ' +level+ ' level')
    plt.savefig(graphics_folder+row_identifier+'_'+labels[i]+'.png')
    i+=1
    scatter_comparison(feats_umap_macenko[phil_indexes_sampled,:2], feats_umap_macenko[xr_indexes_sampled,:2], 'Macenko '+SOURCE_DOMAIN, 'Macenko '+TARGET_DOMAIN, UMAP_domain_list[f] +' UMAP - ' +level+ ' level')
    plt.savefig(graphics_folder+row_identifier+'_'+labels[i]+'.png')
    i+=1
    scatter_comparison(feats_umap[phil_indexes_sampled,:2], feats_umap_lut[phil_indexes_sampled,:2], 'Crude '+SOURCE_DOMAIN, 'LUT '+SOURCE_DOMAIN, UMAP_domain_list[f] +' UMAP - ' +level+ ' level')
    plt.savefig(graphics_folder+row_identifier+'_'+labels[i]+'.png')
    i+=1
    scatter_comparison(feats_umap[xr_indexes_sampled,:2], feats_umap_lut[xr_indexes_sampled,:2], 'Crude '+TARGET_DOMAIN, 'LUT '+TARGET_DOMAIN, UMAP_domain_list[f] +' UMAP - ' +level+ ' level')
    plt.savefig(graphics_folder+row_identifier+'_'+labels[i]+'.png')

    plt.close('all')
    


# df = pd.DataFrame(rows, columns=labels)
# df['identifier'] = row_identifier_list
# df.to_csv(graphics_folder+'UMAP_comparison.csv')

# print(df)


# In[ ]:


#PCA
print('Running PCA')

# In[ ]:


UMAP_domain_list = ['Source', 'LUT', 'MACENKO']
level_list = ['Tile', 'WSI']
# rows = []
# row_identifier_list = []
labels = ['crude_vs_crude',
        'LUT_' + SOURCE_DOMAIN + '_vs_crude_' + TARGET_DOMAIN,
        'LUT_' + SOURCE_DOMAIN +'_vs_' + TARGET_DOMAIN,
        'Macenko_S_vs_T',
        'Source_LUT_vs_Crude',
        'Target_LUT_vs_Crude'
        ]
for fg, training_feat_set in enumerate([tile_feat_sampled, tile_feat_sampled_lut, tile_feat_sampled_macenko, wsi_feats, wsi_feat_lut, wsi_feat_macenko]):
    f=fg%3
    level = level_list[fg//3]
    row_identifier = level+'_PCA_'+UMAP_domain_list[f] 
    row_identifier_list.append(row_identifier)
    print('PCA DOMAIN: '+UMAP_domain_list[f])
    # reducer = umap.UMAP(n_components=10, random_state=42)
    # reducer.fit(training_feat_set) #4min with 0.2 
    reducer = PCA(n_components=10)
    reducer.fit(training_feat_set)

    feats_umap = reducer.transform(tile_feat_sampled) # 12 min
    feats_umap_lut = reducer.transform(tile_feat_sampled_lut) # 20 min both > 38 min
    feats_umap_macenko = reducer.transform(tile_feat_sampled_macenko) # 23 min

    metrics = np.zeros(6)

    i=0
    value = wasserstein_distance_sum(feats_umap[phil_indexes_sampled,:], feats_umap[xr_indexes_sampled,:])
    metrics[i]= value
    i+=1
    # print('crude vs crude: {:.2f}'.format(value)) #9min
    value = wasserstein_distance_sum(feats_umap_lut[phil_indexes_sampled,:], feats_umap[xr_indexes_sampled,:])
    metrics[i]= value
    i+=1
    # print('LUT ' + SOURCE_DOMAIN + ' vs crude XR: {:.2f}'.format(value))
    value = wasserstein_distance_sum(feats_umap_lut[phil_indexes_sampled,:], feats_umap_lut[xr_indexes_sampled,:])
    metrics[i]= value
    i+=1
    # print('LUT ' + SOURCE_DOMAIN+ ' vs LUT XR: {:.2f}'.format(value))
    value = wasserstein_distance_sum(feats_umap_macenko[phil_indexes_sampled,:], feats_umap_macenko[xr_indexes_sampled,:])
    metrics[i]= value
    i+=1
    # print('MACENKO ' + SOURCE_DOMAIN+ ' vs XR: {:.2f}'.format(value))
    value = wasserstein_distance_sum(feats_umap_lut[phil_indexes_sampled,:], feats_umap[phil_indexes_sampled,:])
    metrics[i]= value
    i+=1
    # print(SOURCE_DOMAIN +' LUT vs crude: {:.2f}'.format(value))
    value = wasserstein_distance_sum(feats_umap_lut[xr_indexes_sampled,:], feats_umap[xr_indexes_sampled,:])
    metrics[i]= value
    i+=1
    # print('XR LUT vs crude: {:.2f}'.format(value))

    print(labels)
    print(metrics)
    rows.append(metrics)
    # [print('{:.2f}'.format(m), end=',') for m in metrics]

    for m in metrics: 
        print('{:.2f}'.format(m), end=',')

    i=0
    scatter_comparison(feats_umap[phil_indexes_sampled,:2], feats_umap[xr_indexes_sampled,:2], 'Crude '+SOURCE_DOMAIN, 'Crude '+TARGET_DOMAIN, UMAP_domain_list[f] +' PCA - ' +level+ ' level')
    plt.savefig(graphics_folder+row_identifier+'_'+labels[i]+'.png')
    i+=1
    scatter_comparison(feats_umap_lut[phil_indexes_sampled,:2], feats_umap[xr_indexes_sampled,:2], 'LUT '+SOURCE_DOMAIN, 'Crude '+TARGET_DOMAIN, UMAP_domain_list[f] +' PCA - ' +level+ ' level')
    plt.savefig(graphics_folder+row_identifier+'_'+labels[i]+'.png')
    i+=1
    scatter_comparison(feats_umap_lut[phil_indexes_sampled,:2], feats_umap_lut[xr_indexes_sampled,:2], 'LUT '+SOURCE_DOMAIN, 'LUT '+TARGET_DOMAIN, UMAP_domain_list[f] +' PCA - ' +level+ ' level')
    plt.savefig(graphics_folder+row_identifier+'_'+labels[i]+'.png')
    i+=1
    scatter_comparison(feats_umap_macenko[phil_indexes_sampled,:2], feats_umap_macenko[xr_indexes_sampled,:2], 'Macenko '+SOURCE_DOMAIN, 'Macenko '+TARGET_DOMAIN, UMAP_domain_list[f] +' PCA - ' +level+ ' level')
    plt.savefig(graphics_folder+row_identifier+'_'+labels[i]+'.png')
    i+=1
    scatter_comparison(feats_umap[phil_indexes_sampled,:2], feats_umap_lut[phil_indexes_sampled,:2], 'Crude '+SOURCE_DOMAIN, 'LUT '+SOURCE_DOMAIN, UMAP_domain_list[f] +' PCA - ' +level+ ' level')
    plt.savefig(graphics_folder+row_identifier+'_'+labels[i]+'.png')
    i+=1
    scatter_comparison(feats_umap[xr_indexes_sampled,:2], feats_umap_lut[xr_indexes_sampled,:2], 'Crude '+TARGET_DOMAIN, 'LUT '+TARGET_DOMAIN, UMAP_domain_list[f] +' PCA - ' +level+ ' level')
    plt.savefig(graphics_folder+row_identifier+'_'+labels[i]+'.png')
    
    plt.close('all')

df = pd.DataFrame(rows, columns=labels)
df['identifier'] = row_identifier_list
df.to_csv(graphics_folder+'Feature_comparison.csv')

print(df)


# In[78]:


print('COMPLETED')

