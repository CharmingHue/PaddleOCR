#! /usr/bin
file=${1}

if [ "$file" = "base_en" ]; then
    wget https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_base_en_train.tar 
    tar xf rec_svtr_cppd_base_en_train.tar
    rm rec_svtr_cppd_base_en_train.tar
elif [ "$file" = "base_u14m" ]; then
    wget https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_base_u14m_train.tar
    tar xf rec_svtr_cppd_base_u14m_train.tar
    rm rec_svtr_cppd_base_u14m_train.tar
elif [ "$file" = "tiny_en" ]; then
    wget https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_tiny_en_train.tar
    tar xf rec_svtr_cppd_tiny_en_train.tar
    rm rec_svtr_cppd_tiny_en_train.tar
elif [ "$file" = "base_48_160_en" ]; then
    wget https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_base_48_160_en_train.tar
    tar xf rec_svtr_cppd_base_48_160_en_train.tar
    rm rec_svtr_cppd_base_48_160_en_train.tar
else
    echo "missing necessary parameter ["tiny_en", "base_en", "base_48_160_en", "base_u14m"] "
fi