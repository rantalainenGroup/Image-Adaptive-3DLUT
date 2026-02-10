import argparse
import sys
import torch
import torchvision
import os
from os.path import join as j_
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms
import timm
from utils import *
import time
sys.path.append('/home/bojing/unsupervised_feature_evaluation/model_evaluation/scripts/UNI/')
sys.path.append('/home/bojing/unsupervised_feature_evaluation/model_evaluation/scripts/')


# loading all packages here to start
from uni.downstream.extract_patch_features import *
from dataset import DatasetTiles
from model_factory import *


##### Main #######
parser = argparse.ArgumentParser(description='UNI Inference on external dataset.')
parser.add_argument('--dataset_name',         dest='dataset_name',         type=str,            default=None,        help='Dataset name.')
parser.add_argument('--tile_path',  dest='tile_path',  type=str, default='tile_path_copy', help= 'col to tile_path')
parser.add_argument('--tile_name',  dest='tile_name',  type=str, default='tile_name', help='name of the tiles')

parser.add_argument('--data_out_path',        dest='data_out_path',        type=str,            default=None,        help='Output path.')
parser.add_argument('--df_path',    dest='df_path',    type=str,            default=None,       help='path of external df with all the tiles.')
parser.add_argument('--checkpoint',  dest='checkpoint',  type=str, default=None, required=True,     help='checkpoint required for all models except resnet pretrained with imagenet')
parser.add_argument('--batch_size',  dest='batch_size',  type=int, default=None,       help='Batch size.')
parser.add_argument('--df_feature_name',  dest='df_feature_name',  type=str, default=None,       help='name of the df with the projected features')
parser.add_argument('--model_name',  dest='model_name',  type=str, default=None, help='name of the pretrained model: uni, retcll, ctranspath, resnet18, resnet18_1024')
parser.add_argument('--pretrained', action='store_true', help='Flag to indicate if a pretrained model should be used')
parser.add_argument('--tile_size',  dest='tile_size',  type=int, default=224,  help='Tile size.')
parser.add_argument('--out_dim',  dest='out_dim',  type=int, default=1024, help='Output feature dimension, UNI:1024, resnet18:512')


args               = parser.parse_args()
dataset_name       = args.dataset_name
tile_path          = args.tile_path
tile_name          = args.tile_name
data_out_path      = args.data_out_path
df_path            = args.df_path
checkpoint         = args.checkpoint
batch_size         = args.batch_size
df_feature_name    = args.df_feature_name
model_name         = args.model_name
pretrained         = args.pretrained
tile_size          = args.tile_size
out_dim            = args.out_dim


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# call model 
factory = ModelFactory(model_name, pretrained, checkpoint)
model = factory.create_model()

model.eval()
model.to(device)


# Enable multi-GPU support using DataParallel
print(f"Number of available GPUs: {torch.cuda.device_count()}")
if torch.cuda.device_count() > 1:
    #print(f"Using {torch.cuda.device_count()} GPUs for extraction.")
    #model = torchs.nn.DataParallel(model)
    print("Using 4 GPUs for extraction.")
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])

transform = transforms.Compose(
    [   
        transforms.ToTensor(),  # uint8 -> [0,1]; float stays as-is
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)  # standard for uni model. can double check again


# define otput path
path = os.path.join(data_out_path, model_name, dataset_name, "h"+str(tile_size)+"_w"+str(tile_size)+"_zdim"+str(out_dim))

if not os.path.isdir(path):
    os.makedirs(path)

# load df to project
df = load_df(df_path)
print(f"{len(df)} tiles to process")

# Automatically determine the number of CPU cores
num_workers = os.cpu_count()-4
external_dataset = DatasetTiles(df[tile_path], df[tile_name], transform=transform)
external_dataloader = torch.utils.data.DataLoader(external_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)   


output_path = path + '/' + df_feature_name 

# time the projection process
start_time = time.time()
# extract features
df_feature = extract_patch_features_from_dataloader_custom(model, external_dataloader)
#df_feature = extract_patch_features_from_dataloader(model, external_dataloader)

# Save the DataFrame to pickle file
df_feature.to_pickle(output_path + '.pkl')

# End timing
end_time = time.time()
elapsed_time = (end_time - start_time)/60
print(f"Time taken to generate embeddings: {elapsed_time:.2f} minutes for {len(df)} obs")



        