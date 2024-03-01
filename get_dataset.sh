#!/bin/sh
source=${1:-"None"}

if command -v sudo >/dev/null 2>&1; then
    echo "\033[32msudo is installed\033[0m"
else
    echo "\033[34minstalling sudo\033[0m"
    apt-get update
    apt-get install sudo
fi

if command -v unzip >/dev/null 2>&1; then
    echo "\033[32munzip is installed\033[0m"
else
    echo "\033[34minstalling unzip\033[0m"
    sudo apt-get update
    sudo apt-get install -y unzip
fi

mkdir train_data
cd train_data
if [ "$source" = "huawei" ]; then
    echo "Downloading dataset from Huaweicloud..."
    sudo wget -O train_val_test_lmdb.zip "https://charminghue.obs.cn-north-4.myhuaweicloud.com:443/LMDB/train_val_test_lmdb.zip?AccessKeyId=PHZJJHINZ3VQJ0RBZ8XI&Expires=1708673676&Signature=ouAtGXw7xtSwzBvBIgKaL0lC%2BA4%3D"
    unzip train_val_test_lmdb.zip
    sudo sh extract-datasets.sh
    rm test.zip train.zip val.zip
    cd ..
    rm -rf train_data/train/real/OpenVINO
elif [ "$source" = "directlink" ]; then
    echo "Downloading dataset from DirectLink......"
    sudo wget -O train_val_test_lmdb.zip "http://a18299133.cosfiles.com/a18299133/train_val_test_lmdb.zip"
    unzip train_val_test_lmdb.zip
    sudo sh extract-datasets.sh
    rm test.zip train.zip val.zip
    cd ..
    rm -rf train_data/train/real/OpenVINO
else
    echo  "\033[31m[Error]\033[0m Please Verify Download Source ["huawei", "directlink"]."
fi