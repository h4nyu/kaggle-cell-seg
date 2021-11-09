#!/bin/bash
if [ ! -d /store ]; then
    mkdir /store
fi
cd /store

if [ ! -f sartorius-cell-instance-segmentation.zip ]; then
    kaggle competitions download -c sartorius-cell-instance-segmentation
fi
unzip sartorius-cell-instance-segmentation.zip
rm sartorius-cell-instance-segmentation.zip
