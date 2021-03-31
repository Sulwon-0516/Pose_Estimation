#!/bin/bash
DIR_im="./coco/images"
DIR_anno="./coco"

if [ ! -d "$DIR_im" ];then
    mkdir "$DIR_im"
fi
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d "$DIR_im"

if [ ! -d "$DIR_anno" ];then
    mkdir "$DIR_anno"
fi
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d "$DIR_anno"
