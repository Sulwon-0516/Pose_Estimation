#!/bin/bash
DIR_im="./coco/images"
DIR_anno="./coco"

if [ ! -d "$DIR_im" ];then
    mkdir "$DIR_im"
fi
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d "$DIR_im"

wget http://images.cocodataset.org/zips/test2017.zip
unzip test2017.zip -d "$DIR_im"

if [ ! -d "$DIR_anno" ];then
    mkdir "$DIR_anno"
fi
wget http://images.cocodataset.org/annotations/image_info_test2017.zip
unzip image_info_test2017.zip -d "$DIR_anno"
