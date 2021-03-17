python3 main_few_shot.py \
--batch_size  128 \
--ckpt_dir /root/data/mentali/IDS/ckpt/siamese_base_ce \
--is_tuning False \
--train_file_path /root/data/mentali/IDS/data/few_shot/train.txt \
--dev_file_path /root/data/mentali/IDS/data/few_shot/val.txt \
--params_path /root/data/mentali/IDS/param/hierachical_cnn_params.json \



python3 main_few_shot.py \
--batch_size  128 \
--ckpt_dir /root/data/mentali/IDS/ckpt/siamese_base_ce_tuning \
--is_tuning True \
--restore_ckpt /root/data/mentali/IDS/ckpt/hierarchical_cnn/hierarchical_cnn_1 \
--train_file_path /root/data/mentali/IDS/data/few_shot/train.txt \
--dev_file_path /root/data/mentali/IDS/data/few_shot/val.txt \
--params_path /root/data/mentali/IDS/param/hierachical_cnn_params.json \



python3 main_few_shot.py \
--batch_size  128 \
--ckpt_dir /root/data/mentali/IDS/ckpt/siamese_base_ce_noise \
--is_tuning False \
--train_file_path /root/data/mentali/IDS/data/few_shot/train_noise.txt \
--dev_file_path /root/data/mentali/IDS/data/few_shot/val_noise.txt \
--params_path /root/data/mentali/IDS/param/hierachical_cnn_params.json \