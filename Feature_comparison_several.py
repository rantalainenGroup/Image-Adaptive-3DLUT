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


#Get all features:
usecols = ["tile_name", "scanner_model_new", 'macenko_blur', 'crude_tile_path','png_tile_path','macenko_tile_path']


# EXP_ID = '20'
# SOURCE_DOMAIN = 'PHILIPS'
# TARGET_DOMAIN = 'XR'
# base_folder = '/mnt/ssd/ferbue/Image-Adaptive-3DLUT/LUTs/unpaired/exp_'+ EXP_ID +'/'
# feature_folder = base_folder + 'uni/scanb_malmo/h224_w224_zdim1024/'
# feat_csv = feature_folder + 'scanb_malmo_source_exp20_bothscans_uni.pkl' # 1742609 rows
# feat_lut_csv = feature_folder + 'scanb_malmo_lut_exp20_bothscans_uni.pkl' #LUT Both philips and XR #1742609 rows
# feat_macenko_csv = feature_folder + 'scanb_malmo_macenko_bothscans_uni.pkl' #LUT Both philips and XR #1742609 rows
# test_df_path = '/mnt/ssd/ferbue/Image-Adaptive-3DLUT/dataframes/lut_exp20_bothscanners_v2.csv'

EXP_ID="26" # change
SOURCE_DOMAIN="APERIO" # change
TARGET_DOMAIN="XR" # change

base_folder = '/mnt/ssd2/ferbue/Image-Adaptive-3DLUT/LUTs/unpaired/exp_'+ EXP_ID +'/'
feature_folder = base_folder + 'uni/scanb_malmo/h224_w224_zdim1024/'

feat_csv=feature_folder +"scanb_malmo_source_all_uni.pkl" # change
feat_lut_csv=feature_folder +"scanb_malmo_lut_all_uni.pkl" # change
feat_macenko_csv=feature_folder +"scanb_malmo_macenko_all_uni.pkl" # change
test_df_path="/mnt/ssd/ferbue/Image-Adaptive-3DLUT/dataframes/lut_exp23_allscanners_v2.csv" # change



graphics_folder = base_folder + 'feat_comparison/'
if not os.path.isdir(graphics_folder):
    os.mkdir(graphics_folder)

print('Reading all the stuff')
    
test_df = pd.read_csv(test_df_path, usecols=usecols) #1742609 rows

features = pd.read_pickle(feat_csv)
features2 = pd.merge(features, test_df, on="tile_name") #to get the scanner_model linked to the features
feats = features2.drop(columns=usecols).to_numpy() #numerical feats only

Source_list = features2.scanner_model_new.unique()

#list of indexes for each scanner
# phil_indexes = (features2['scanner_model']==SOURCE_DOMAIN).to_list()
# xr_indexes = [not elem for elem in phil_indexes]
# xr1_idxs = (features2['scanner_model']=='XR1').to_list()
# xr2_idxs = (features2['scanner_model']=='XR2').to_list()

#LUT features,
features_lut = pd.read_pickle(feat_lut_csv)
features_lut_scanner = pd.merge(features_lut, features2[['tile_name','scanner_model_new']], on="tile_name")
feat_lut= features_lut_scanner.drop(columns=['tile_name','scanner_model_new']).to_numpy()

#MACENKO features,
features_macenko = pd.read_pickle(feat_macenko_csv)
features_macenko_scanner = pd.merge(features_macenko, features2[['tile_name','scanner_model_new']], on="tile_name")
feat_macenko= features_macenko_scanner.drop(columns=['tile_name','scanner_model_new']).to_numpy()

idx_by_source = {}
for source_domain in Source_list:
    idx_by_source[source_domain] = (features2['scanner_model_new']==source_domain).to_list()


#WSI aggregation
features2['file_name'] = features2['tile_name'].str.split('_clean').str[0]
df_grouped = (
    features2.groupby('file_name')
      .agg(lambda x: x.mean() if x.dtype != 'object' else x.iloc[0])
      .reset_index()
)

wsi_feats = df_grouped.drop(columns=usecols +['file_name']).to_numpy()
print(wsi_feats.shape) # N, 1024

features_lut_scanner['file_name'] = features_lut_scanner['tile_name'].str.split('_clean').str[0]
grouped_lut = (
    features_lut_scanner.groupby('file_name')
      .agg(lambda x: x.mean() if x.dtype != 'object' else x.iloc[0])
      .reset_index()
)
wsi_feat_lut = grouped_lut.drop(columns=['file_name','tile_name','scanner_model_new']).to_numpy()
print(wsi_feat_lut.shape)

features_macenko_scanner['file_name'] = features_macenko_scanner['tile_name'].str.split('_clean').str[0]
grouped_macenko = (
    features_macenko_scanner.groupby('file_name')
      .agg(lambda x: x.mean() if x.dtype != 'object' else x.iloc[0])
      .reset_index()
)
wsi_feat_macenko = grouped_macenko.drop(columns=['file_name','tile_name','scanner_model_new']).to_numpy()
print(wsi_feat_macenko.shape)


WSI_source_idxs = {}
for source_domain in Source_list:
    WSI_source_idxs[source_domain] = (df_grouped['scanner_model_new']==source_domain).to_list()

rows = []
row_identifier_list = []
labels = ['crude vs crude','LUT vs crude', 'LUT vs LUT', 'Macenko']
level_list = ['Tile', 'WSI']

#########################################################################################################################

print('Running Raw')
for level in level_list:
    print(level)
    
    if level == 'Tile': #
        feats_umap = feats
        feats_umap_lut = feat_lut
        feats_umap_macenko = feat_macenko

        source_idxs = idx_by_source

    else: #WSI
        feats_umap = wsi_feats
        feats_umap_lut = wsi_feat_lut
        feats_umap_macenko = wsi_feat_macenko

        source_idxs = WSI_source_idxs

    for source_domain in Source_list:
        print(source_domain)
        row_identifier = level+'_UNI_'+ source_domain
        row_identifier_list.append(row_identifier)
        metrics = np.zeros(4)
        #crude S vs crude T
        metrics[0] = wasserstein_distance_sum(feats_umap[source_idxs[source_domain]], feats_umap[source_idxs[TARGET_DOMAIN]])
        #LUT S vs crude T
        metrics[1] = wasserstein_distance_sum(feats_umap_lut[source_idxs[source_domain]], feats_umap[source_idxs[TARGET_DOMAIN]])
        #LUT s vs LUT T
        metrics[2] = wasserstein_distance_sum(feats_umap_lut[source_idxs[source_domain]], feats_umap_lut[source_idxs[TARGET_DOMAIN]])
        #Macenko
        metrics[3] = wasserstein_distance_sum(feats_umap_macenko[source_idxs[source_domain]], feats_umap_macenko[source_idxs[TARGET_DOMAIN]])
        rows.append(metrics)

        
#########################################################################################################################

print('Running UMAP')
for level in level_list:

    if level == 'Tile': #

        reducer = umap.UMAP(n_components=10, random_state=42)
        reducer.fit(feats) #4min with 0.2 

        feats_umap = reducer.transform(feats) # 12 min
        feats_umap_lut = reducer.transform(feat_lut) # 20 min both > 38 min
        feats_umap_macenko = reducer.transform(feat_macenko) # 23 min

        source_idxs = idx_by_source

    else: #WSI
        reducer = umap.UMAP(n_components=10, random_state=42)
        reducer.fit(wsi_feats) #4min with 0.2 
    
        feats_umap = reducer.transform(wsi_feats) # 12 min
        feats_umap_lut = reducer.transform(wsi_feat_lut) # 20 min both > 38 min
        feats_umap_macenko = reducer.transform(wsi_feat_macenko) # 23 min

        source_idxs = WSI_source_idxs

    for source_domain in Source_list:
        row_identifier = level+'_UMAP_'+ source_domain
        row_identifier_list.append(row_identifier)
        metrics = np.zeros(4)
        #crude S vs crude T
        metrics[0] = wasserstein_distance_sum(feats_umap[source_idxs[source_domain]], feats_umap[source_idxs[TARGET_DOMAIN]])
        #LUT S vs crude T
        metrics[1] = wasserstein_distance_sum(feats_umap_lut[source_idxs[source_domain]], feats_umap[source_idxs[TARGET_DOMAIN]])
        #LUT s vs LUT T
        metrics[2] = wasserstein_distance_sum(feats_umap_lut[source_idxs[source_domain]], feats_umap_lut[source_idxs[TARGET_DOMAIN]])
        #Macenko
        metrics[3] = wasserstein_distance_sum(feats_umap_macenko[source_idxs[source_domain]], feats_umap_macenko[source_idxs[TARGET_DOMAIN]])
        rows.append(metrics)

        scatter_comparison(feats_umap[source_idxs[source_domain],:2], feats_umap[source_idxs[TARGET_DOMAIN],:2], 'Crude '+source_domain, 'Crude '+TARGET_DOMAIN, 'UMAP - ' +level+ ' level')
        plt.savefig(graphics_folder+row_identifier+'_'+'crude_vs_crude'+'.png')

        scatter_comparison(feats_umap_lut[source_idxs[source_domain],:2], feats_umap[source_idxs[TARGET_DOMAIN],:2], 'LUT '+source_domain, 'Crude '+TARGET_DOMAIN, 'UMAP - ' +level+ ' level')
        plt.savefig(graphics_folder+row_identifier+'_'+'LUT_S_crude_T'+'.png')

        scatter_comparison(feats_umap_lut[source_idxs[source_domain],:2], feats_umap_lut[source_idxs[TARGET_DOMAIN],:2], 'LUT '+source_domain, 'LUT '+TARGET_DOMAIN, 'UMAP - ' +level+ ' level')
        plt.savefig(graphics_folder+row_identifier+'_'+'LUT_S_T'+'.png')

        scatter_comparison(feats_umap_macenko[source_idxs[source_domain],:2], feats_umap_macenko[source_idxs[TARGET_DOMAIN],:2], 'MAC '+source_domain, 'MAC '+TARGET_DOMAIN, 'UMAP - ' +level+ ' level')
        plt.savefig(graphics_folder+row_identifier+'_'+'Macenko_S_T'+'.png')

        plt.close('all')



#########################################################################################################################

print('Running PCA')
for level in level_list:

    if level == 'Tile': #

        reducer = PCA(n_components=10)
        reducer.fit(feats) #4min with 0.2 

        feats_umap = reducer.transform(feats) # 12 min
        feats_umap_lut = reducer.transform(feat_lut) # 20 min both > 38 min
        feats_umap_macenko = reducer.transform(feat_macenko) # 23 min

        source_idxs = idx_by_source

    else: #WSI
        reducer = PCA(n_components=10)
        reducer.fit(wsi_feats) #4min with 0.2 
    
        feats_umap = reducer.transform(wsi_feats) # 12 min
        feats_umap_lut = reducer.transform(wsi_feat_lut) # 20 min both > 38 min
        feats_umap_macenko = reducer.transform(wsi_feat_macenko) # 23 min

        source_idxs = WSI_source_idxs

    for source_domain in Source_list:
        row_identifier = level+'_PCA_'+ source_domain
        row_identifier_list.append(row_identifier)
        metrics = np.zeros(4)
        #crude S vs crude T
        metrics[0] = wasserstein_distance_sum(feats_umap[source_idxs[source_domain]], feats_umap[source_idxs[TARGET_DOMAIN]])
        #LUT S vs crude T
        metrics[1] = wasserstein_distance_sum(feats_umap_lut[source_idxs[source_domain]], feats_umap[source_idxs[TARGET_DOMAIN]])
        #LUT s vs LUT T
        metrics[2] = wasserstein_distance_sum(feats_umap_lut[source_idxs[source_domain]], feats_umap_lut[source_idxs[TARGET_DOMAIN]])
        #Macenko
        metrics[3] = wasserstein_distance_sum(feats_umap_macenko[source_idxs[source_domain]], feats_umap_macenko[source_idxs[TARGET_DOMAIN]])
        rows.append(metrics)

        scatter_comparison(feats_umap[source_idxs[source_domain],:2], feats_umap[source_idxs[TARGET_DOMAIN],:2], 'Crude '+source_domain, 'Crude '+TARGET_DOMAIN, 'UMAP - ' +level+ ' level')
        plt.savefig(graphics_folder+row_identifier+'_'+'crude_vs_crude'+'.png')

        scatter_comparison(feats_umap_lut[source_idxs[source_domain],:2], feats_umap[source_idxs[TARGET_DOMAIN],:2], 'LUT '+source_domain, 'Crude '+TARGET_DOMAIN, 'UMAP - ' +level+ ' level')
        plt.savefig(graphics_folder+row_identifier+'_'+'LUT_S_crude_T'+'.png')

        scatter_comparison(feats_umap_lut[source_idxs[source_domain],:2], feats_umap_lut[source_idxs[TARGET_DOMAIN],:2], 'LUT '+source_domain, 'LUT '+TARGET_DOMAIN, 'UMAP - ' +level+ ' level')
        plt.savefig(graphics_folder+row_identifier+'_'+'LUT_S_T'+'.png')

        scatter_comparison(feats_umap_macenko[source_idxs[source_domain],:2], feats_umap_macenko[source_idxs[TARGET_DOMAIN],:2], 'MAC '+source_domain, 'MAC '+TARGET_DOMAIN, 'UMAP - ' +level+ ' level')
        plt.savefig(graphics_folder+row_identifier+'_'+'Macenko_S_T'+'.png')

        plt.close('all')

df = pd.DataFrame(rows, columns=labels)
df['identifier'] = row_identifier_list
df.to_csv(graphics_folder+'Feature_comparison.csv')

print(df)



######################################### SECOND APROACH
# Second approach, all at once LUT, one by one UMAP
rows = []
row_identifier_list = []
labels = ['crude vs crude','LUT vs crude', 'LUT vs LUT', 'Macenko']
level_list = ['Tile', 'WSI']
print('Second approach')
print('Running UMAP')
for source_domain in Source_list:
    print(source_domain)
    for level in level_list:

        if level == 'Tile': #
            source_idxs = idx_by_source
            reducer = umap.UMAP(n_components=10, random_state=42)
            reducer.fit(np.concatenate([feats[source_idxs[source_domain]], feats[source_idxs[TARGET_DOMAIN]]])) #4min with 0.2 

            feats_umap = reducer.transform(feats) # 12 min
            feats_umap_lut = reducer.transform(feat_lut) # 20 min both > 38 min
            feats_umap_macenko = reducer.transform(feat_macenko) # 23 min

            

        else: #WSI
            source_idxs = WSI_source_idxs
            reducer = umap.UMAP(n_components=10, random_state=42)
            reducer.fit(np.concatenate([wsi_feats[source_idxs[source_domain]], wsi_feats[source_idxs[TARGET_DOMAIN]]])) #4min with 0.2 
        
            feats_umap = reducer.transform(wsi_feats) # 12 min
            feats_umap_lut = reducer.transform(wsi_feat_lut) # 20 min both > 38 min
            feats_umap_macenko = reducer.transform(wsi_feat_macenko) # 23 min

            

        # for source_domain in Source_list:
        row_identifier = level+'_UMAP_'+ source_domain
        row_identifier_list.append(row_identifier)
        metrics = np.zeros(4)
        #crude S vs crude T
        metrics[0] = wasserstein_distance_sum(feats_umap[source_idxs[source_domain]], feats_umap[source_idxs[TARGET_DOMAIN]])
        #LUT S vs crude T
        metrics[1] = wasserstein_distance_sum(feats_umap_lut[source_idxs[source_domain]], feats_umap[source_idxs[TARGET_DOMAIN]])
        #LUT s vs LUT T
        metrics[2] = wasserstein_distance_sum(feats_umap_lut[source_idxs[source_domain]], feats_umap_lut[source_idxs[TARGET_DOMAIN]])
        #Macenko
        metrics[3] = wasserstein_distance_sum(feats_umap_macenko[source_idxs[source_domain]], feats_umap_macenko[source_idxs[TARGET_DOMAIN]])
        rows.append(metrics)

# rows = []
# row_identifier_list = []
# labels = ['crude vs crude','LUT vs crude', 'LUT vs LUT', 'Macenko']
# level_list = ['Tile', 'WSI']

print('Running PCA')
for source_domain in Source_list:
    print(source_domain)
    for level in level_list:

        if level == 'Tile': #
            source_idxs = idx_by_source
            reducer = PCA(n_components=10)
            reducer.fit(np.concatenate([feats[source_idxs[source_domain]], feats[source_idxs[TARGET_DOMAIN]]])) #4min with 0.2 

            feats_umap = reducer.transform(feats) # 12 min
            feats_umap_lut = reducer.transform(feat_lut) # 20 min both > 38 min
            feats_umap_macenko = reducer.transform(feat_macenko) # 23 min

            

        else: #WSI
            source_idxs = WSI_source_idxs
            reducer = PCA(n_components=10)
            reducer.fit(np.concatenate([wsi_feats[source_idxs[source_domain]], wsi_feats[source_idxs[TARGET_DOMAIN]]])) #4min with 0.2 
        
            feats_umap = reducer.transform(wsi_feats) # 12 min
            feats_umap_lut = reducer.transform(wsi_feat_lut) # 20 min both > 38 min
            feats_umap_macenko = reducer.transform(wsi_feat_macenko) # 23 min

            

        # for source_domain in Source_list:
        row_identifier = level+'_PCA_'+ source_domain
        row_identifier_list.append(row_identifier)
        metrics = np.zeros(4)
        #crude S vs crude T
        metrics[0] = wasserstein_distance_sum(feats_umap[source_idxs[source_domain]], feats_umap[source_idxs[TARGET_DOMAIN]])
        #LUT S vs crude T
        metrics[1] = wasserstein_distance_sum(feats_umap_lut[source_idxs[source_domain]], feats_umap[source_idxs[TARGET_DOMAIN]])
        #LUT s vs LUT T
        metrics[2] = wasserstein_distance_sum(feats_umap_lut[source_idxs[source_domain]], feats_umap_lut[source_idxs[TARGET_DOMAIN]])
        #Macenko
        metrics[3] = wasserstein_distance_sum(feats_umap_macenko[source_idxs[source_domain]], feats_umap_macenko[source_idxs[TARGET_DOMAIN]])
        rows.append(metrics)


df = pd.DataFrame(rows, columns=labels)
df['identifier'] = row_identifier_list
df.to_csv(graphics_folder+'Feature_comparison_byScan.csv')















# In[78]:


print('COMPLETED')
