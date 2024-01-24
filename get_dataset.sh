#!/bin/sh
mkdir train_data && cd train_data
wget -O train_val_test_lmdb.zip "https://charminghue.obs.cn-north-4.myhuaweicloud.com:443/LMDB/train_val_test_lmdb.zip?AccessKeyId=PHZJJHINZ3VQJ0RBZ8XI&Expires=1708673676&Signature=ouAtGXw7xtSwzBvBIgKaL0lC%2BA4%3D"
unzip train_val_test_lmdb.zip
sh extract-datasets.sh
rm test.zip train.zip val.zip
cd ..
rm -rf train_data/train/real/OpenVINO