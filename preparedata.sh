#/bin/bash

git clone -b AutoDL https://github.com/CharmingHue/PaddleOCR.git
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
cd PaddleOCR 
pip install -r requirements.txt 
sh get_dataset.sh directlink
