#!/bin/bash
DIR_im="/home/PoseEstimation/coco/images"
DIR_anno="/home/PoseEstimation/coco"

if [ ! -d "$DIR_im" ];then
    mkdir "$DIR_im"
    wget http://images.cocodataset.org/zips/val2017.zip
    unzip val2017.zip -d "$DIR_im"
fi
if [ ! -d "$DIR_anno" ];then
    mkdir "$DIR_anno"
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip annotations_trainval2017.zip -d "$DIR_anno"
fi