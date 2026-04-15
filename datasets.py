import glob
import random
import os
import numpy as np
import torch
import cv2

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision_x_functional as TF_x
import pandas as pd
from PIL import Image
import platform


class ImageDataset_sRGB(Dataset):
    def __init__(self, root, mode="train", unpaird_data="fiveK", combined=True):
        self.mode = mode
        self.unpaird_data = unpaird_data

        file = open(os.path.join(root,'train_input.txt'),'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root,"input","JPG/480p",set1_input_files[i][:-1] + ".jpg"))
            self.set1_expert_files.append(os.path.join(root,"expertC","JPG/480p",set1_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'train_label.txt'),'r')
        set2_input_files = sorted(file.readlines())
        self.set2_input_files = list()
        self.set2_expert_files = list()
        for i in range(len(set2_input_files)):
            self.set2_input_files.append(os.path.join(root,"input","JPG/480p",set2_input_files[i][:-1] + ".jpg"))
            self.set2_expert_files.append(os.path.join(root,"expertC","JPG/480p",set2_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'test.txt'),'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root,"input","JPG/480p",test_input_files[i][:-1] + ".jpg"))
            self.test_expert_files.append(os.path.join(root,"expertC","JPG/480p",test_input_files[i][:-1] + ".jpg"))

        if combined:
            self.set1_input_files = self.set1_input_files + self.set2_input_files
            self.set1_expert_files = self.set1_expert_files + self.set2_expert_files


    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = Image.open(self.set1_input_files[index % len(self.set1_input_files)])
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = Image.open(self.test_input_files[index % len(self.test_input_files)])
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])

        if self.mode == "train":

            ratio_H = np.random.uniform(0.6,1.0)
            ratio_W = np.random.uniform(0.6,1.0)
            W,H = img_input._size
            crop_h = round(H*ratio_H)
            crop_w = round(W*ratio_W)
            i, j, h, w = transforms.RandomCrop.get_params(img_input, output_size=(crop_h, crop_w))
            img_input = TF.crop(img_input, i, j, h, w)
            img_exptC = TF.crop(img_exptC, i, j, h, w)
            #img_input = TF.resized_crop(img_input, i, j, h, w, (320,320))
            #img_exptC = TF.resized_crop(img_exptC, i, j, h, w, (320,320))

            if np.random.random() > 0.5:
                img_input = TF.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            a = np.random.uniform(0.8,1.2)
            img_input = TF.adjust_brightness(img_input,a)

            a = np.random.uniform(0.8,1.2)
            img_input = TF.adjust_saturation(img_input,a)

        img_input = TF.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)

        return {"A_input": img_input, "A_exptC": img_exptC, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)


class ImageDataset_XYZ(Dataset):
    def __init__(self, root, mode="train", unpaird_data="fiveK", combined=True):
        self.mode = mode

        file = open(os.path.join(root,'train_input.txt'),'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root,"input","PNG/480p_16bits_XYZ_WB",set1_input_files[i][:-1] + ".png"))
            self.set1_expert_files.append(os.path.join(root,"expertC","JPG/480p",set1_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'train_label.txt'),'r')
        set2_input_files = sorted(file.readlines())
        self.set2_input_files = list()
        self.set2_expert_files = list()
        for i in range(len(set2_input_files)):
            self.set2_input_files.append(os.path.join(root,"input","PNG/480p_16bits_XYZ_WB",set2_input_files[i][:-1] + ".png"))
            self.set2_expert_files.append(os.path.join(root,"expertC","JPG/480p",set2_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'test.txt'),'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root,"input","PNG/480p_16bits_XYZ_WB",test_input_files[i][:-1] + ".png"))
            self.test_expert_files.append(os.path.join(root,"expertC","JPG/480p",test_input_files[i][:-1] + ".jpg"))

        if combined:
            self.set1_input_files = self.set1_input_files + self.set2_input_files
            self.set1_expert_files = self.set1_expert_files + self.set2_expert_files


    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = cv2.imread(self.set1_input_files[index % len(self.set1_input_files)],-1)
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)],-1)
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])

        img_input = np.array(img_input)
        #img_input = np.array(cv2.cvtColor(img_input,cv2.COLOR_BGR2RGB))

        if self.mode == "train":

            ratio_H = np.random.uniform(0.6,1.0)
            ratio_W = np.random.uniform(0.6,1.0)
            W,H = img_exptC._size
            crop_h = round(H*ratio_H)
            crop_w = round(W*ratio_W)
            i, j, h, w = transforms.RandomCrop.get_params(img_exptC, output_size=(crop_h, crop_w))
            img_input = TF_x.crop(img_input, i, j, h, w)
            img_exptC = TF.crop(img_exptC, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF_x.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            a = np.random.uniform(0.6,1.4)
            img_input = TF_x.adjust_brightness(img_input,a)

        img_input = TF_x.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)

        return {"A_input": img_input, "A_exptC": img_exptC, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)

class ImageDataset_sRGB_unpaired(Dataset):
    def __init__(self, root, mode="train", unpaird_data="fiveK"):
        self.mode = mode
        self.unpaird_data = unpaird_data

        file = open(os.path.join(root,'train_input.txt'),'r')  # source domain
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root,"input","JPG/480p",set1_input_files[i][:-1] + ".jpg")) # input in source domain
            self.set1_expert_files.append(os.path.join(root,"expertC","JPG/480p",set1_input_files[i][:-1] + ".jpg")) # paired data in the target domain, only used to calculated psnr

        file = open(os.path.join(root,'train_label.txt'),'r')
        set2_input_files = sorted(file.readlines())
        self.set2_input_files = list()
        self.set2_expert_files = list()
        for i in range(len(set2_input_files)):
            self.set2_input_files.append(os.path.join(root,"input","JPG/480p",set2_input_files[i][:-1] + ".jpg")) # paried data in source domain, not used
            self.set2_expert_files.append(os.path.join(root,"expertC","JPG/480p",set2_input_files[i][:-1] + ".jpg")) # target domain

        file = open(os.path.join(root,'test.txt'),'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root,"input","JPG/480p",test_input_files[i][:-1] + ".jpg"))
            self.test_expert_files.append(os.path.join(root,"expertC","JPG/480p",test_input_files[i][:-1] + ".jpg"))


    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = Image.open(self.set1_input_files[index % len(self.set1_input_files)])
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])
            seed = random.randint(1,len(self.set2_expert_files))
            img2 = Image.open(self.set2_expert_files[(index + seed) % len(self.set2_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = Image.open(self.test_input_files[index % len(self.test_input_files)])
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])
            img2 = img_exptC

        if self.mode == "train":
            ratio_H = np.random.uniform(0.6,1.0)
            ratio_W = np.random.uniform(0.6,1.0)
            W,H = img_input._size
            crop_h = round(H*ratio_H)
            crop_w = round(W*ratio_W)
            W2,H2 = img2._size
            crop_h = min(crop_h,H2)
            crop_w = min(crop_w,W2)
            i, j, h, w = transforms.RandomCrop.get_params(img_input, output_size=(crop_h, crop_w))
            img_input = TF.crop(img_input, i, j, h, w)
            img_exptC = TF.crop(img_exptC, i, j, h, w)
            i, j, h, w = transforms.RandomCrop.get_params(img2, output_size=(crop_h, crop_w))
            img2 = TF.crop(img2, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            if np.random.random() > 0.5:
                img2 = TF.hflip(img2)

            #if np.random.random() > 0.5:
            #    img_input = TF.vflip(img_input)
            #    img_exptC = TF.vflip(img_exptC)
            #    img2 = TF.vflip(img2)

            a = np.random.uniform(0.6,1.4)
            img_input = TF.adjust_brightness(img_input,a)

            a = np.random.uniform(0.8,1.2)
            img_input = TF.adjust_saturation(img_input,a)


        img_input = TF.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)
        img2 = TF.to_tensor(img2)

        return {"A_input": img_input, "A_exptC": img_exptC, "B_exptC": img2, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)

# adde dataset.py to allow feeding a .csv file

def load_acc(path_img):
    return Image(path_img)

def load_PIL(path_img):
    return Image.open(path_img)

load_acc = load_PIL

class ImageDataset_sRGB_unpaired_CSV(Dataset):
    """
    CSV must have: path (str), scanner_model_new in {"PHILIPS","XR"}, split in {"train","test"}.
    """
    def __init__(self, csv_path, mode="train", test_domain="PHILIPS",
                 col_name='crude_tile_path', data_augmentation=False):
        assert mode in {"train","test"}
        assert test_domain in {"PHILIPS","XR",'S360'}
        self.mode = mode
        self.test_domain = test_domain
        self.data_augmentation = data_augmentation
        if platform.system() == 'Linux':
            self._load = load_acc
        else:
            self._load = load_PIL

        # Fast CSV read & filter
        usecols = [col_name, "scanner_model_new", "split", "blur"]
        df = pd.read_csv(csv_path, usecols=usecols, dtype={col_name:str, "scanner_model_new":str, "split":str})
        df = df.dropna(subset=[col_name, "scanner_model_new", "split"]).copy()
        df["path"] = df[col_name].astype(str).str.strip()
        df = df[df["blur"]>250]

        # Vectorized partitioning (no iterrows)
        mA = (df["scanner_model_new"]=="PHILIPS")
        mB = (df["scanner_model_new"]=="XR")
        mTr = (df["split"]=="train")
        mTe = (df["split"]=="test")

        self.A_train = df.loc[mA & mTr, "path"].tolist()
        self.B_train = df.loc[mB & mTr, "path"].tolist()
        self.A_test  = df.loc[mA & mTe, "path"].tolist()
        self.B_test  = df.loc[mB & mTe, "path"].tolist()

        print('Train samples:',len(self.A_train), len(self.B_train))
        print('Test samples:',len(self.A_test), len(self.B_test))

        if mode=="train":
            if not self.A_train or not self.B_train:
                raise RuntimeError("Need non-empty PHILIPS and XR TRAIN sets.")
        else:
            pool = self.A_test if test_domain=="PHILIPS" else self.B_test
            if not pool:
                raise RuntimeError(f"Empty {test_domain} TEST set.")

    def __len__(self):
        if self.mode == "train":
            return len(self.A_train)
        else:
            return len(self.A_test if self.test_domain=="PHILIPS" else self.B_test)    

    def __getitem__(self, index):
        if self.mode == "train":
            a_path = self.A_train[index % len(self.A_train)]
            img_input = self._load(a_path)
            img_exptC = self._load(a_path)  # identity placeholder

            # random XR for unpaired B
            seed_idx = random.randint(0, len(self.B_train)-1)
            img2 = self._load(self.B_train[seed_idx])
            
            """
            # --- Do not need to crop  ---
            ratio_H = np.random.uniform(0.6,1.0)
            ratio_W = np.random.uniform(0.6,1.0)
            W,H = img_input.size
            crop_h = round(H*ratio_H); crop_w = round(W*ratio_W)
            W2,H2 = img2.size
            crop_h = min(crop_h,H2); crop_w = min(crop_w,W2)
            i, j, h, w = transforms.RandomCrop.get_params(img_input, output_size=(crop_h, crop_w))
            img_input = TF.crop(img_input, i, j, h, w)
            img_exptC = TF.crop(img_exptC, i, j, h, w)
            i2, j2, h2, w2 = transforms.RandomCrop.get_params(img2, output_size=(crop_h, crop_w))
            img2 = TF.crop(img2, i2, j2, h2, w2)
            """
            # --- same augmentations as original ---

            if self.data_augmentation and self.mode=="train":
                if np.random.random() > 0.5:
                    img_input = TF.hflip(img_input); img_exptC = TF.hflip(img_exptC)
                if np.random.random() > 0.5:
                    img2 = TF.hflip(img2)

                a = np.random.uniform(0.6,1.4); img_input = TF.adjust_brightness(img_input,a)
                a = np.random.uniform(0.8,1.2); img_input = TF.adjust_saturation(img_input,a)
            # only get one domain 
        elif self.mode == "test":
            pool = self.A_test if self.test_domain=="PHILIPS" else self.B_test
            img_input = self._load(pool[index % len(pool)])
            img_exptC = img_input
            img2 = img_exptC
        
        # to tensor [0,1]
        return {
            "A_input": TF.to_tensor(img_input),
            "A_exptC": TF.to_tensor(img_exptC),
            "B_exptC": TF.to_tensor(img2),
            "input_name": os.path.basename(self.A_train[index % len(self.A_train)]) if self.mode=="train"
                          else os.path.basename((self.A_test if self.test_domain=='PHILIPS' else self.B_test)[index % len(pool)])
        }
    
class ImageDataset_sRGB_unpaired_CSV_v2(Dataset):
    """
    CSV must have: path (str), scanner_model_new in {"PHILIPS","XR"}, split in {"train","test"}.
    """
    def __init__(self, csv_path, mode="train", source_domain="PHILIPS", target_domain='XR',
                 col_name='crude_tile_path', data_augmentation=False):
        assert mode in {"train","test"}
        assert source_domain in {"PHILIPS","XR",'S360',"APERIO"}
        assert target_domain in {"PHILIPS","XR",'S360',"APERIO"}
        self.mode = mode
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.data_augmentation = data_augmentation
        if platform.system() == 'Linux':
            self._load = load_acc
        else:
            self._load = load_PIL

        # Fast CSV read & filter
        usecols = [col_name, "scanner_model_new", "split", "blur"]
        df = pd.read_csv(csv_path, usecols=usecols, dtype={col_name:str, "scanner_model_new":str, "split":str})
        df = df.dropna(subset=[col_name, "scanner_model_new", "split"]).copy()
        df["path"] = df[col_name].astype(str).str.strip()
        df = df[df["blur"]>250]

        # Vectorized partitioning (no iterrows)
        mA = (df["scanner_model_new"]==source_domain)
        mB = (df["scanner_model_new"]==target_domain)
        mTr = (df["split"]=="train")
        mTe = (df["split"]=="test")

        self.A_train = df.loc[mA & mTr, "path"].tolist()
        self.B_train = df.loc[mB & mTr, "path"].tolist()
        self.A_test  = df.loc[mA & mTe, "path"].tolist()
        self.B_test  = df.loc[mB & mTe, "path"].tolist()

        print('Train samples:',len(self.A_train), len(self.B_train))
        print('Test samples:',len(self.A_test), len(self.B_test))

        if mode=="train":
            if not self.A_train or not self.B_train:
                raise RuntimeError("Need non-empty PHILIPS and XR TRAIN sets.")
        else:
            pool = self.A_test #if source_domain=="PHILIPS" else self.B_test
            if not pool:
                raise RuntimeError(f"Empty {source_domain} TEST set.")

    def __len__(self):
        if self.mode == "train":
            return len(self.A_train)
        else:
            return len(self.A_test)  #if self.test_domain=="PHILIPS" else self.B_test  

    def __getitem__(self, index):
        if self.mode == "train":
            a_path = self.A_train[index % len(self.A_train)]
            img_input = self._load(a_path)
            img_exptC = self._load(a_path)  # identity placeholder

            # random XR for unpaired B
            seed_idx = random.randint(0, len(self.B_train)-1)
            img2 = self._load(self.B_train[seed_idx])
            
            """
            # --- Do not need to crop  ---
            ratio_H = np.random.uniform(0.6,1.0)
            ratio_W = np.random.uniform(0.6,1.0)
            W,H = img_input.size
            crop_h = round(H*ratio_H); crop_w = round(W*ratio_W)
            W2,H2 = img2.size
            crop_h = min(crop_h,H2); crop_w = min(crop_w,W2)
            i, j, h, w = transforms.RandomCrop.get_params(img_input, output_size=(crop_h, crop_w))
            img_input = TF.crop(img_input, i, j, h, w)
            img_exptC = TF.crop(img_exptC, i, j, h, w)
            i2, j2, h2, w2 = transforms.RandomCrop.get_params(img2, output_size=(crop_h, crop_w))
            img2 = TF.crop(img2, i2, j2, h2, w2)
            """
            # --- same augmentations as original ---

            if self.data_augmentation and self.mode=="train":
                if np.random.random() > 0.5:
                    img_input = TF.hflip(img_input); img_exptC = TF.hflip(img_exptC)
                if np.random.random() > 0.5:
                    img2 = TF.hflip(img2)

                a = np.random.uniform(0.6,1.4); img_input = TF.adjust_brightness(img_input,a)
                a = np.random.uniform(0.8,1.2); img_input = TF.adjust_saturation(img_input,a)
            # only get one domain 
        elif self.mode == "test":
            pool = self.A_test # if self.test_domain=="PHILIPS" else self.B_test
            img_input = self._load(pool[index % len(pool)])
            img_exptC = img_input
            img2 = img_exptC
        
        # to tensor [0,1]
        return {
            "A_input": TF.to_tensor(img_input),
            "A_exptC": TF.to_tensor(img_exptC),
            "B_exptC": TF.to_tensor(img2),
            "input_name": os.path.basename(self.A_train[index % len(self.A_train)]) if self.mode=="train"
                          else os.path.basename((self.A_test)[index % len(pool)]) # within (self.A_test if self.test_domain=='PHILIPS' else self.B_test)
        }

class ImageDataset_sRGB_unpaired_CSV_several(Dataset):
    """
    CSV must have: path (str), scanner_model_new in {"PHILIPS","XR"}, split in {"train","test"}.
    """
    def __init__(self, csv_path, mode="train", source_domain_list=["PHILIPS"], target_domain='XR',
                 col_name='crude_tile_path', data_augmentation=False):
        assert mode in {"train","test"}
        assert all(s in {"PHILIPS", "XR", "S360", "APERIO"} for s in source_domain_list)
        assert target_domain in {"PHILIPS","XR",'S360',"APERIO"}
        self.mode = mode
        self.source_domain_list = source_domain_list
        self.target_domain = target_domain
        self.data_augmentation = data_augmentation
        if platform.system() == 'Linux':
            self._load = load_acc
        else:
            self._load = load_PIL

        # Fast CSV read & filter
        usecols = [col_name, "scanner_model_new", "split", "blur"]
        df = pd.read_csv(csv_path, usecols=usecols, dtype={col_name:str, "scanner_model_new":str, "split":str})
        df = df.dropna(subset=[col_name, "scanner_model_new", "split"]).copy()
        df["path"] = df[col_name].astype(str).str.strip()
        df = df[df["blur"]>250]

        # Vectorized partitioning (no iterrows)
        mA = (df["scanner_model_new"].isin(source_domain_list))
        mB = (df["scanner_model_new"]==target_domain)
        mTr = (df["split"]=="train")
        mTe = (df["split"]=="test")

        self.A_train = df.loc[mA & mTr, "path"].tolist()
        self.B_train = df.loc[mB & mTr, "path"].tolist()
        self.A_test  = df.loc[mA & mTe, "path"].tolist()
        self.B_test  = df.loc[mB & mTe, "path"].tolist()

        print('Train samples:',len(self.A_train), len(self.B_train))
        print('Test samples:',len(self.A_test), len(self.B_test))

        if mode=="train":
            if not self.A_train or not self.B_train:
                raise RuntimeError("Need non-empty PHILIPS and XR TRAIN sets.")
        else:
            pool = self.A_test #if source_domain=="PHILIPS" else self.B_test
            if not pool:
                raise RuntimeError(f"Empty {source_domain_list} TEST set.")

    def __len__(self):
        if self.mode == "train":
            return len(self.A_train)
        else:
            return len(self.A_test)  #if self.test_domain=="PHILIPS" else self.B_test  

    def __getitem__(self, index):
        if self.mode == "train":
            a_path = self.A_train[index % len(self.A_train)]
            img_input = self._load(a_path)
            img_exptC = self._load(a_path)  # identity placeholder

            # random XR for unpaired B
            seed_idx = random.randint(0, len(self.B_train)-1)
            img2 = self._load(self.B_train[seed_idx])
            
            """
            # --- Do not need to crop  ---
            ratio_H = np.random.uniform(0.6,1.0)
            ratio_W = np.random.uniform(0.6,1.0)
            W,H = img_input.size
            crop_h = round(H*ratio_H); crop_w = round(W*ratio_W)
            W2,H2 = img2.size
            crop_h = min(crop_h,H2); crop_w = min(crop_w,W2)
            i, j, h, w = transforms.RandomCrop.get_params(img_input, output_size=(crop_h, crop_w))
            img_input = TF.crop(img_input, i, j, h, w)
            img_exptC = TF.crop(img_exptC, i, j, h, w)
            i2, j2, h2, w2 = transforms.RandomCrop.get_params(img2, output_size=(crop_h, crop_w))
            img2 = TF.crop(img2, i2, j2, h2, w2)
            """
            # --- same augmentations as original ---

            if self.data_augmentation and self.mode=="train":
                if np.random.random() > 0.5:
                    img_input = TF.hflip(img_input); img_exptC = TF.hflip(img_exptC)
                if np.random.random() > 0.5:
                    img2 = TF.hflip(img2)

                a = np.random.uniform(0.6,1.4); img_input = TF.adjust_brightness(img_input,a)
                a = np.random.uniform(0.8,1.2); img_input = TF.adjust_saturation(img_input,a)
            # only get one domain 
        elif self.mode == "test":
            pool = self.A_test # if self.test_domain=="PHILIPS" else self.B_test
            img_input = self._load(pool[index % len(pool)])
            img_exptC = img_input
            img2 = img_exptC
        
        # to tensor [0,1]
        return {
            "A_input": TF.to_tensor(img_input),
            "A_exptC": TF.to_tensor(img_exptC),
            "B_exptC": TF.to_tensor(img2),
            "input_name": os.path.basename(self.A_train[index % len(self.A_train)]) if self.mode=="train"
                          else os.path.basename((self.A_test)[index % len(pool)]) # within (self.A_test if self.test_domain=='PHILIPS' else self.B_test)
        }

   
class ImageDataset_sRGB_unpaired_CSV_inference(Dataset):
    """
    CSV must have: file_name, crude_tile_path, scanner_model_new in {PHILIPS, XR}, split in {train, test}.
    Pass a per-WSI df (one WSI at a time).
    """
    def __init__(self, df, mode="test", test_domain="PHILIPS",
                 file_name="file_name", col_name="crude_tile_path"):
        assert test_domain in {"PHILIPS", "XR"}
        self.mode = mode
        self.test_domain = test_domain
        self._file_name = str(df[file_name].iloc[0])  # WSI id for all samples in this dataset

        self._load = load_acc if platform.system() == "Linux" else load_PIL

        # Normalize path column
        df = df.copy()
        df["path"] = df[col_name].astype(str).str.strip()

        # Partition
        mA  = (df["scanner_model_new"] == "PHILIPS")
        mB  = (df["scanner_model_new"] == "XR")
        mTe = (df["split"] == "test")

        self.A_test = df.loc[mA & mTe, "path"].tolist()
        self.B_test = df.loc[mB & mTe, "path"].tolist()

        pool = self.A_test if test_domain == "PHILIPS" else self.B_test
        if not pool:
            raise RuntimeError(f"Empty {test_domain} TEST set.")

    def __len__(self):
        return len(self.A_test if self.test_domain == "PHILIPS" else self.B_test)

    def __getitem__(self, index):
        if self.mode == "test":
            pool = self.A_test if self.test_domain == "PHILIPS" else self.B_test
            p = pool[index % len(pool)] # This result is always an integer in [0, len(pool)-1].
            img_input = self._load(p)
            img_exptC = img_input
            img2 = img_exptC
        else:
            raise RuntimeError("Inference dataset expects mode='test'.")

        return {
            "A_input": TF.to_tensor(img_input),
            "A_exptC": TF.to_tensor(img_exptC),
            "B_exptC": TF.to_tensor(img2),
            "input_name": os.path.basename(p),
            "file_name": self._file_name,   # WSI-specific id for grouping/aggregation
        }
    


class ImageDataset_sRGB_unpaired_CSV_inference_v2(Dataset):
    """
    CSV must have: file_name, crude_tile_path, scanner_model_new in {PHILIPS, XR}, split in {train, test}.
    Pass a per-WSI df (one WSI at a time).
    """
    def __init__(self, df, mode="test", source_domain="PHILIPS", target_domain='XR',
                 file_name="file_name", col_name="crude_tile_path"):
        assert source_domain in {"PHILIPS", "XR", "S360","APERIO"}
        assert target_domain in {"PHILIPS", "XR", "S360","APERIO"}
        self.mode = mode
        self.source_domain = source_domain
        self.target_domain = target_domain
        self._file_name = str(df[file_name].iloc[0])  # WSI id for all samples in this dataset

        self._load = load_acc if platform.system() == "Linux" else load_PIL

        # Normalize path column
        df = df.copy()
        df["path"] = df[col_name].astype(str).str.strip()

        # Partition
        mA  = (df["scanner_model_new"] == self.source_domain)
        mB  = (df["scanner_model_new"] == self.target_domain)
        mTe = (df["split"] == "test")

        self.A_test = df.loc[mA & mTe, "path"].tolist()
        self.B_test = df.loc[mB & mTe, "path"].tolist()

        pool = self.A_test #if test_domain == "PHILIPS" else self.B_test
        if not pool:
            raise RuntimeError(f"Empty {self.source_domain} TEST set.")

    def __len__(self):
        return len(self.A_test ) #if self.test_domain == "PHILIPS" else self.B_test

    def __getitem__(self, index):
        if self.mode == "test":
            pool = self.A_test #if self.test_domain == "PHILIPS" else self.B_test
            p = pool[index % len(pool)] # This result is always an integer in [0, len(pool)-1].
            img_input = self._load(p)
            img_exptC = img_input
            img2 = img_exptC
        else:
            raise RuntimeError("Inference dataset expects mode='test'.")

        return {
            "A_input": TF.to_tensor(img_input),
            "A_exptC": TF.to_tensor(img_exptC),
            "B_exptC": TF.to_tensor(img2),
            "input_name": os.path.basename(p),
            "file_name": self._file_name,   # WSI-specific id for grouping/aggregation
        }

class ImageDataset_sRGB_unpaired_CSV_inference_several(Dataset):
    """
    CSV must have: file_name, crude_tile_path, scanner_model_new in {PHILIPS, XR}, split in {train, test}.
    Pass a per-WSI df (one WSI at a time).
    """
    def __init__(self, df, mode="test", source_domain_list=["PHILIPS"], target_domain='XR',
                 file_name="file_name", col_name="crude_tile_path"):
        assert all(s in {"PHILIPS", "XR", "S360","APERIO"} for s in source_domain_list)
        assert target_domain in {"PHILIPS", "XR", "S360","APERIO"}
        self.mode = mode
        self.source_domain_list = source_domain_list
        self.target_domain = target_domain
        self._file_name = str(df[file_name].iloc[0])  # WSI id for all samples in this dataset

        self._load = load_acc if platform.system() == "Linux" else load_PIL

        # Normalize path column
        df = df.copy()
        df["path"] = df[col_name].astype(str).str.strip()

        # Partition
        mA  = (df["scanner_model_new"].isin(self.source_domain_list))
        mB  = (df["scanner_model_new"] == self.target_domain)
        mTe = (df["split"] == "test")

        self.A_test = df.loc[mA & mTe, "path"].tolist()
        self.B_test = df.loc[mB & mTe, "path"].tolist()

        pool = self.A_test #if test_domain == "PHILIPS" else self.B_test
        if not pool:
            raise RuntimeError(f"Empty {self.source_domain_list} TEST set.")

    def __len__(self):
        return len(self.A_test ) #if self.test_domain == "PHILIPS" else self.B_test

    def __getitem__(self, index):
        if self.mode == "test":
            pool = self.A_test #if self.test_domain == "PHILIPS" else self.B_test
            p = pool[index % len(pool)] # This result is always an integer in [0, len(pool)-1].
            img_input = self._load(p)
            img_exptC = img_input
            img2 = img_exptC
        else:
            raise RuntimeError("Inference dataset expects mode='test'.")

        return {
            "A_input": TF.to_tensor(img_input),
            "A_exptC": TF.to_tensor(img_exptC),
            "B_exptC": TF.to_tensor(img2),
            "input_name": os.path.basename(p),
            "file_name": self._file_name,   # WSI-specific id for grouping/aggregation
        }




class ImageDataset_XYZ_unpaired(Dataset):
    def __init__(self, root, mode="train", unpaird_data="fiveK"):
        self.mode = mode
        self.unpaird_data = unpaird_data

        file = open(os.path.join(root,'train_input.txt'),'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root,"input","PNG/480p_16bits_XYZ_WB",set1_input_files[i][:-1] + ".png"))
            self.set1_expert_files.append(os.path.join(root,"expertC","JPG/480p",set1_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'train_label.txt'),'r')
        set2_input_files = sorted(file.readlines())
        self.set2_input_files = list()
        self.set2_expert_files = list()
        for i in range(len(set2_input_files)):
            self.set2_input_files.append(os.path.join(root,"input","PNG/480p_16bits_XYZ_WB",set2_input_files[i][:-1] + ".png"))
            self.set2_expert_files.append(os.path.join(root,"expertC","JPG/480p",set2_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'test.txt'),'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root,"input","PNG/480p_16bits_XYZ_WB",test_input_files[i][:-1] + ".png"))
            self.test_expert_files.append(os.path.join(root,"expertC","JPG/480p",test_input_files[i][:-1] + ".jpg"))


    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = cv2.imread(self.set1_input_files[index % len(self.set1_input_files)],-1)
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])
            seed = random.randint(1,len(self.set2_expert_files))
            img2 = Image.open(self.set2_expert_files[(index + seed) % len(self.set2_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)],-1)
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])
            img2 = img_exptC

        img_input = np.array(img_input)
        #img_input = np.array(cv2.cvtColor(img_input,cv2.COLOR_BGR2RGB))

        if self.mode == "train":
            ratio_H = np.random.uniform(0.6,1.0)
            ratio_W = np.random.uniform(0.6,1.0)
            W,H = img_exptC._size
            crop_h = round(H*ratio_H)
            crop_w = round(W*ratio_W)
            W2,H2 = img2._size
            crop_h = min(crop_h,H2)
            crop_w = min(crop_w,W2)
            i, j, h, w = transforms.RandomCrop.get_params(img_exptC, output_size=(crop_h, crop_w))
            img_input = TF_x.crop(img_input, i, j, h, w)
            img_exptC = TF.crop(img_exptC, i, j, h, w)
            i, j, h, w = transforms.RandomCrop.get_params(img2, output_size=(crop_h, crop_w))
            img2 = TF.crop(img2, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF_x.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            if np.random.random() > 0.5:
                img2 = TF.hflip(img2)

            a = np.random.uniform(0.6,1.4)
            img_input = TF_x.adjust_brightness(img_input,a)

        img_input = TF_x.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)
        img2 = TF.to_tensor(img2)

        return {"A_input": img_input, "A_exptC": img_exptC, "B_exptC": img2, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)


class ImageDataset_HDRplus(Dataset):
    def __init__(self, root, mode="train", combined=True):
        self.mode = mode

        file = open(os.path.join(root,'train.txt'),'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root,"middle_480p",set1_input_files[i][:-1] + ".png"))
            self.set1_expert_files.append(os.path.join(root,"output_480p",set1_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'test.txt'),'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root,"middle_480p",test_input_files[i][:-1] + ".png"))
            self.test_expert_files.append(os.path.join(root,"output_480p",test_input_files[i][:-1] + ".jpg"))


    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = cv2.imread(self.set1_input_files[index % len(self.set1_input_files)],-1)
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)],-1)
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])

        img_input = np.array(img_input)
        #img_input = np.array(cv2.cvtColor(img_input,cv2.COLOR_BGR2RGB))

        if self.mode == "train":

            ratio = np.random.uniform(0.6,1.0)
            W,H = img_exptC._size
            crop_h = round(H*ratio)
            crop_w = round(W*ratio)
            i, j, h, w = transforms.RandomCrop.get_params(img_exptC, output_size=(crop_h, crop_w))
            try:
                img_input = TF_x.crop(img_input, i, j, h, w)
            except:
                print(crop_h,crop_w,img_input.shape())
            img_exptC = TF.crop(img_exptC, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF_x.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            a = np.random.uniform(0.6,1.4)
            img_input = TF_x.adjust_brightness(img_input,a)

            #a = np.random.uniform(0.8,1.2)
            #img_input = TF_x.adjust_saturation(img_input,a)

        img_input = TF_x.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)

        return {"A_input": img_input, "A_exptC": img_exptC, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)

class ImageDataset_HDRplus_unpaired(Dataset):
    def __init__(self, root, mode="train"):
        self.mode = mode

        file = open(os.path.join(root,'train.txt'),'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root,"middle_480p",set1_input_files[i][:-1] + ".png"))
            self.set1_expert_files.append(os.path.join(root,"output_480p",set1_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'train.txt'),'r')
        set2_input_files = sorted(file.readlines())
        self.set2_input_files = list()
        self.set2_expert_files = list()
        for i in range(len(set2_input_files)):
            self.set2_input_files.append(os.path.join(root,"middle_480p",set2_input_files[i][:-1] + ".png"))
            self.set2_expert_files.append(os.path.join(root,"output_480p",set2_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'test.txt'),'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root,"middle_480p",test_input_files[i][:-1] + ".png"))
            self.test_expert_files.append(os.path.join(root,"output_480p",test_input_files[i][:-1] + ".jpg"))


    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = cv2.imread(self.set1_input_files[index % len(self.set1_input_files)],-1)
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])
            seed = random.randint(1,len(self.set2_expert_files))
            img2 = Image.open(self.set2_expert_files[(index + seed) % len(self.set2_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)],-1)
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])
            img2 = img_exptC

        img_input = np.array(img_input)
        #img_input = np.array(cv2.cvtColor(img_input,cv2.COLOR_BGR2RGB))

        if self.mode == "train":
            ratio = np.random.uniform(0.6,1.0)
            W,H = img_exptC._size
            crop_h = round(H*ratio)
            crop_w = round(W*ratio)
            W2,H2 = img2._size
            crop_h = min(crop_h,H2)
            crop_w = min(crop_w,W2)
            i, j, h, w = transforms.RandomCrop.get_params(img_exptC, output_size=(crop_h, crop_w))
            img_input = TF_x.crop(img_input, i, j, h, w)
            img_exptC = TF.crop(img_exptC, i, j, h, w)
            i, j, h, w = transforms.RandomCrop.get_params(img2, output_size=(crop_h, crop_w))
            img2 = TF.crop(img2, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF_x.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            if np.random.random() > 0.5:
                img2 = TF.hflip(img2)

            a = np.random.uniform(0.8,1.2)
            img_input = TF_x.adjust_brightness(img_input,a)

        img_input = TF_x.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)
        img2 = TF.to_tensor(img2)

        return {"A_input": img_input, "A_exptC": img_exptC, "B_exptC": img2, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)
