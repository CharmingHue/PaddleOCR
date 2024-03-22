#!/bin/bash
config_path=${1}
model_path=${2}

# 定义数据集名称数组
# datasets=('IC13_857' 'SVT' 'IIIT5k' 'IC15_1811' 'SVTP' 'CUTE80')
datasets=('artistic' 'contextless' 'curve' 'general' 'multi_oriented' 'multi_words' 'salient')

# 循环遍历数据集数组
for dataset in "${datasets[@]}"
do

    echo "Evaluating dataset ${dataset}..."
    # python tools/eval.py -c $config_path -o Global.pretrained_model=$model_path Eval.dataset.data_dir=train_data/test/${dataset}
    python tools/eval.py -c $config_path -o Global.pretrained_model=$model_path Eval.dataset.data_dir=train_data/u14m/lmdb_format/${dataset}
    # bash ./tools/eval_all.sh ./configs/rec/rec_cloformer_cppd.yml ./output/rec/cloformer_cppd_base/iter_epoch_6.pdparams
done

echo "Evaluation completed."

