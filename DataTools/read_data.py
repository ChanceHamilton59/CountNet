#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 jwilches <jwilches@lambda-dual>
#
# Distributed under terms of the MIT license.

import os
import numpy as np
import matplotlib.pyplot as pt
import argparse as ap
import h5py

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument("input_path", help="input path to save trials")
    args = parser.parse_args()

    file_list = [os.path.join(args.input_path,item) for item in os.listdir(args.input_path) if 'npy' in item]
    print(file_list[0:1])
    
    for item in file_list[0:1]:
        with open(item,'rb') as f:
            timeSeries = np.load(f)
        
        with h5py.File(item[:-4] + '_rgbd.h5','r') as hf:
            rgbdSeries = hf['rgbd'][:]

        with h5py.File(item[:-4] + '_video.h5','r') as hf:
            sideSeries = hf['video'][:]
        
        with h5py.File(item[:-4] + '_video_rgb.h5','r') as hf:
            cupSeries = hf['video'][:]
        
        with h5py.File(item[:-4] + '_video_fixed.h5','r') as hf:
            fixedSeries = hf['video'][:]

        print(timeSeries.shape)
        print(rgbdSeries.shape)
        print(sideSeries.shape)
        print(cupSeries.shape)
        print(fixedSeries.shape)
    
    frame = 200
    pt.figure()
    axList = [pt.subplot(2,3,i) for i in range(1,7)]
    
    #Time series
    axList[0].plot(timeSeries)
    axList[0].set_title('Time Series')
    #RGBD of camera rotating together with the cup
    axList[1].imshow(rgbdSeries[frame,...,0:3].astype(np.uint8))
    axList[1].set_title('RGB Cropped')
    axList[2].imshow(rgbdSeries[frame,...,3])
    axList[2].set_title('Depth Cropped')
    #RGBD of side, looking inside the cup (no cropping), or fixed
    axList[3].imshow(sideSeries[frame,...])
    axList[3].set_title('Side Camera')
    axList[4].imshow(cupSeries[frame,...])
    axList[4].set_title('Cup Camera')
    axList[5].imshow(fixedSeries[frame,...])
    axList[5].set_title('Fixed Camera')

    pt.tight_layout()
    pt.show()

