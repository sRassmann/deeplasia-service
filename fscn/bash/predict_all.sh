#! /bin/bash
target="/home/rassman/bone2gene/hand-segmentation/FSCNN/output/masks"
for filename in rsna_bone_age/bone_age_training_data_set; do
#for filename in ACh  healthy_magd  HyCh  Noonan  PsHPT  SGA_Magd  shox_magdeburg  SRS_Magd  uts_leipzig  uts_magdeburg; do
    mkdir "$target/$filename"
    python ../predict.py --input /home/rassman/bone2gene/data/annotated/$filename --use_gpu \
     --checkpoint /home/rassman/bone2gene/hand-segmentation/FSCNN/output/pretrained_tensormask_cosine/ckp/best_model.ckpt \
    --output_dir "/home/rassman/bone2gene/data/masks/fscnn_cos/$filename"
#    --output_dir "$target/$filename"
done
