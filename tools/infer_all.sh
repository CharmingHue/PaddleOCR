#!/bin/bash
config_path=${1}
model_path=${2}
method=${3}
# 定义数据集名称数组
# datasets=('IC13' 'SVT' 'IIIT5K' 'IC15' 'SVTP' 'CUTE80')
# datasets=('artistic' 'contextless' 'curve' 'multi_oriented' 'multi_words' 'salient')
datasets=( 'general')

# 循环遍历数据集数组
for dataset in "${datasets[@]}"
do

    echo "Infer dataset ${dataset}..."
    if [ "$dataset" = "general" ]; then
        # 使用find命令获取目录列表
        sub_dirs=$(find /root/autodl-tmp/Union14M-L/full_images/ -maxdepth 1 -mindepth 1 -type d)
        echo "$sub_dirs"
        # 如果需要对每个子目录进行操作，可以再次遍历$sub_dirs
        for dir in $sub_dirs; do
            if [[ "$dir" = "/root/autodl-tmp/Union14M-L/full_images/art_scene" ]]; then
                echo "处理目录：$dir"
                python tools/infer_rec.py -c $config_path -o Global.pretrained_model=$model_path Global.infer_img=$dir/imgs Global.save_res_path=./output/rec/predicts_${method}_${dataset}.txt
            # 这里可以添加对每个子目录进行的操作
            fi
        done
    fi
    python tools/infer_rec.py -c $config_path -o Global.pretrained_model=$model_path Global.infer_img=/root/autodl-tmp/Union14M-L/Union14M-Benchmarks/${dataset}/imgs/ Global.save_res_path=./output/rec/predicts_${method}_${dataset}.txt
    # python tools/infer_rec.py -c $config_path -o Global.pretrained_model=$model_path Global.infer_img=./train_data/common_benchmarks/${dataset}/imgs/ Global.save_res_path=./output/rec/predicts_${method}_${dataset}.txt
    # python tools/eval.py -c $config_path -o Global.pretrained_model=$model_path Eval.dataset.data_dir=train_data/u14m/lmdb_format/${dataset}
    # bash ./tools/eval_all.sh ./configs/rec/rec_cloformer_cppd.yml ./output/rec/cloformer_cppd_base/iter_epoch_6.pdparams
done

echo "Evaluation completed."

