python main.py \
--batch_size 128 \
--model_name cnn \
--cnn_modules vgg \
--num_epochs 20 \
--is_early_stop False \
--train_file_path /root/data/mentali/IDS/data/cnn/train_file_compress_all.txt \
--dev_file_path /root/data/mentali/IDS/data/cnn/test_file_compress_all.txt \
--label_mapping_path /root/data/mentali/IDS/cache/label_mapping_all.pkl \
--params_path /root/data/mentali/IDS/param/cnn_params.json \

python main.py \
--batch_size 128 \
--model_name cnn \
--cnn_modules resnet \
--num_epochs 20 \
--is_early_stop False \
--train_file_path /root/data/mentali/IDS/data/cnn/train_file_compress_all.txt \
--dev_file_path /root/data/mentali/IDS/data/cnn/test_file_compress_all.txt \
--label_mapping_path /root/data/mentali/IDS/cache/label_mapping_all.pkl \
--params_path /root/data/mentali/IDS/param/cnn_params.json \

python main.py \
--batch_size 128 \
--model_name cnn \
--cnn_modules mobile_net \
--num_epochs 20 \
--is_early_stop False \
--train_file_path /root/data/mentali/IDS/data/cnn/train_file_compress_all.txt \
--dev_file_path /root/data/mentali/IDS/data/cnn/test_file_compress_all.txt \
--label_mapping_path /root/data/mentali/IDS/cache/label_mapping_all.pkl \
--params_path /root/data/mentali/IDS/param/cnn_params.json \



