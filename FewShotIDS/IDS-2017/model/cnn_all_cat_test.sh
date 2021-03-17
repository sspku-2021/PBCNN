python main.py \
--batch_size 128 \
--model_name cnn \
--cnn_modules vgg \
--mode test \
--test_file_path /root/data/mentali/IDS/data/cnn/test_file_compress_all.txt \
--label_mapping_path /root/data/mentali/IDS/cache/label_mapping_all.pkl \
--params_path /root/data/mentali/IDS/param/cnn_params.json \
--restore_ckpt /root/data/mentali/IDS/ckpt/cnn_vgg/cnn_vgg-16672 \
--file_output /root/data/mentali/IDS/output/vgg_all.txt \

python main.py \
--batch_size 128 \
--model_name cnn \
--cnn_modules resnet \
--mode test \
--test_file_path /root/data/mentali/IDS/data/cnn/test_file_compress_all.txt \
--label_mapping_path /root/data/mentali/IDS/cache/label_mapping_all.pkl \
--params_path /root/data/mentali/IDS/param/cnn_params.json \
--restore_ckpt /root/data/mentali/IDS/ckpt/cnn_resnet/cnn_resnet_4 \
--file_output /root/data/mentali/IDS/output/resnet_all.txt \

python main.py \
--batch_size 128 \
--model_name cnn \
--cnn_modules mobile_net \
--mode test \
--test_file_path /root/data/mentali/IDS/data/cnn/test_file_compress_all.txt \
--label_mapping_path /root/data/mentali/IDS/cache/label_mapping_all.pkl \
--params_path /root/data/mentali/IDS/param/cnn_params.json \
--restore_ckpt /root/data/mentali/IDS/ckpt/cnn_mobile_net/cnn_mobile_net-18756 \
--file_output /root/data/mentali/IDS/output/mobile_all.txt \

