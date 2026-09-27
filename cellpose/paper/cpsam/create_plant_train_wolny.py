import os
import numpy as np
import matplotlib.pyplot as plt 
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from cellpose import io, transforms, utils, models, dynamics, metrics
import h5py
import gc
from glob import glob
import cv2
from natsort import natsorted 
import shutil
from pathlib import Path
import pandas as pd 
from tqdm import trange
import fastremap
from skimage.measure import label


def create_plant_train(lateral_root = True):
    """ Data from Wolny et al 2020 https://osf.io/uzq3w/overview
    
    For lateral root we used Movie2 only for testing since it was fully segmented

    """
    if lateral_root:
        root = Path("/media/carsen/ssd3/plants/lateral_root/")
        # 0.1625 × 0.1625 × 0.250 um/px
        anisotropy = 0.25 / 0.1625
        rsz = 0.25
    else:
        root = Path("/media/carsen/ssd3/plants/ovules/")
        # 0.075 × 0.075 × 0.235 um/px
        anisotropy = 0.235 / 0.075
        rsz = 0.75
    print(anisotropy)

    ly, lx = 256, 256
    pm = [(0,1,2), (1,0,2), (2,0,1)]
    pstr = ["YX", "ZX", "ZY"]

    train_files = (root / "train").glob("*.h5")
    train_files = natsorted([tf for tf in train_files])
    test_files = (root / "test").glob("*.h5")
    test_files = natsorted([tf for tf in test_files])
    print(len(train_files), len(test_files))

    np.random.seed(0)
    for k, files in enumerate([train_files, test_files]):
        for i, tf in enumerate(files):
            print(tf.stem)
            f = h5py.File(tf, "r")
            print(f.keys())
            img = np.array(f["raw"])
            masks = (np.array(f["label"])).astype("uint16")            
            if lateral_root:
                if k==0: masks -= 1
                else:
                    if i==0: masks[masks==1] = 0
                    else: masks[masks==411] = 0
            else:
                ignore = np.array(f["label_with_ignore"])==-1
                zignore = ignore.mean(axis=(1,2))
                zmask = (masks>0).mean(axis=(1,2))
                zmin = np.nonzero(zignore < zmask*0.2)[0]
                if len(zmin) == 0: continue
                zmin = zmin[0]
                zmax = np.nonzero(zignore[zmin:] > zmask[zmin:]*0.2)[0]
                zmax = zmax[0] + zmin if len(zmax) > 0 else len(zmask)
                print(zmin, zmax)
                if zmax - zmin < 50: continue
                masks = masks[zmin:zmax]
                img = img[zmin:zmax]
                
            fastremap.renumber(masks, in_place=True)
            print(f"ncells = {masks.max()}")
            ### resize
            print(img.shape)
            if rsz!=1:
                Lyr = int(masks.shape[-2] * rsz)
                Lxr = int(masks.shape[-1] * rsz)
                masks_rsz = transforms.resize_image(masks, Ly=Lyr, Lx=Lxr, no_channels=True, 
                                                interpolation=cv2.INTER_NEAREST).astype(masks.dtype)
                img_rsz = transforms.resize_image(img, Ly=Lyr, Lx=Lxr, 
                                                no_channels=True).astype(img.dtype)
            else:
                masks_rsz = masks.copy()
                img_rsz = img.copy()
            # make isotropic
            Lyr = int(masks_rsz.shape[0] * anisotropy * rsz)
            Lxr = int(masks_rsz.shape[-1])
            masks_rsz = transforms.resize_image(masks_rsz.transpose(1,0,2), Ly=Lyr, Lx=Lxr, 
                                    no_channels=True, interpolation=cv2.INTER_NEAREST).astype(masks.dtype).transpose(1,0,2)
            img_rsz = transforms.resize_image(img_rsz.transpose(1,0,2), Ly=Lyr, Lx=Lxr, 
                                        no_channels=True).astype(img.dtype).transpose(1,0,2)
            print(img_rsz.shape)
            if lateral_root and k==1:
                masks_rsz = masks_rsz[:130]
                img_rsz = img_rsz[:130]
            else:
                th = 0.05 # fraction of mask pixels required for cropping
                for d, inds in enumerate([(1, 2), (0, 2), (0, 1)]):
                    # compute fraction of mask pixels in each slice
                    m = (masks_rsz > 0).mean(axis=inds)
                    imin = max(0, np.nonzero(m>th)[0][0] - 10)
                    imax = min(len(m), len(m) - np.nonzero(m[::-1]>th)[0][0] + 10)
                    # slice d dimension
                    masks_rsz = masks_rsz.take(range(imin, imax), axis=d)
                    img_rsz = img_rsz.take(range(imin, imax), axis=d)

            img = img_rsz.copy()
            masks = masks_rsz.copy()
            print(img.shape, masks.shape)
            
            if k==1:
                folder = root / "test_3D"
                folder.mkdir(exist_ok=True)
                io.imsave(folder / f"{tf.stem}.tif", img_rsz)
                io.imsave(folder / f"{tf.stem}_masks.tif", masks_rsz)
            
            for p in range(3):
                n_slices = 20 if p==0 else 10
                if p > 0:
                    masks_rsz = masks.transpose(pm[p])
                    img_rsz = img.transpose(pm[p])
                else:
                    masks_rsz = masks.copy() 
                    img_rsz = img.copy()

                Lz, Ly, Lx = masks_rsz.shape
                
                # random z/y/x slices
                #nr = 400
                #iz = np.random.randint(Lz - 2*zpad, size=(nr,)) + zpad 
                zpad = 10
                iz = np.linspace(zpad, Lz - zpad, n_slices*2, dtype=int)
                mr = masks_rsz[iz]
                mp = (mr>0).mean(axis=(-2,-1))
                igood = np.sort(mp.argsort()[::-1][:n_slices])
                mr = mr[igood]
                iz = iz[igood]
                imr = img_rsz[iz]
                mr = [label(mr[i]) for i in range(len(mr))]
                
                # save training crops
                (root / "slices").mkdir(exist_ok=True)
                folder = root / "slices" / "train" if k == 0 else root / "slices" / "test"
                folder.mkdir(exist_ok=True)
                for i in range(len(iz)):
                    M0 = mr[i].copy()
                    Im0 = imr[i]
                    M0 = utils.fill_holes_and_remove_small_masks(M0, 100)
                    fastremap.renumber(M0, in_place=True)
                    io.imsave(folder / f"{tf.stem}_{pstr[p]}_{i:03d}.tif", Im0)
                    io.imsave(folder / f"{tf.stem}_{pstr[p]}_{i:03d}_masks.tif", M0)
