# -*- coding: utf-8 -*-
"""
@author: Lan Wen (lw2773)
"""
# rootSIFT

import numpy as np
import csv
import cv2
from os import listdir
import scipy.cluster.vq as vq
from sklearn.cluster import MiniBatchKMeans

img_path = 'C:/Users/lanwe/Desktop/train/images/'
image_list = listdir(img_path)

image_dir_list = []
for i in range(len(image_list)):
    image_dir_list.append(img_path+image_list[i])

descs =np.empty((0,128))
eps = 1e-7
for i in range(len(image_dir_list)):
    img = cv2.imread(image_dir_list[i])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img_gray,None)
    kp,des = sift.compute(img_gray,kp)
    des /= (des.sum(axis=1, keepdims=True) + eps)
    des = np.sqrt(des)
    descs = np.vstack((descs,des))

n_cluster = 1000
kmeans_sift = MiniBatchKMeans(init='k-means++', n_clusters=n_cluster, batch_size=100,n_init=10, 
                              init_size = 3*n_cluster,max_no_improvement=20, verbose=0,compute_labels= False).fit(descs)
kmeans_center = kmeans_sift.cluster_centers_

def computeHistograms(kmeans_center, descriptors):
    code, dist = vq.vq(descriptors, kmeans_center)
    histogram_of_words, bin_edges = np.histogram(code,
                                                 bins=range(kmeans_center.shape[0] + 1),
                                                 normed=True)
    return histogram_of_words

feature = np.empty((0,n_cluster))
for i in range(len(image_dir_list)):
    img = cv2.imread(image_dir_list[i])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img_gray,None)
    kp,des = sift.compute(img_gray,kp)
    des /= (des.sum(axis=1, keepdims=True) + eps)
    des = np.sqrt(des)
    result = computeHistograms(kmeans_center,des)
    feature = np.vstack((feature,result))

label = np.array(image_list)
label = label.reshape(len(image_list),1)
feature_label = np.concatenate((label, feature), 1)
feature_list = feature_label.tolist()

with open("C:/Users/lanwe/Desktop/rtsift_feature.csv", "w") as f:
    writer = csv.writer(f,lineterminator='\n')
    writer.writerows(feature_list)