python main.py \
--batch_size 128 \
--model_name hierarchical_cnn \
--is_pretrain True \
--is_time True \
--train_file_path /root/data/mentali/IDS/data/cnn/train_file_compress_all.txt \
--dev_file_path /root/data/mentali/IDS/data/cnn/test_file_compress_all.txt \
--params_path /root/data/mentali/IDS/param/hierachical_cnn_params.json \

mv  /data/mentali/IDS/ckpt/hierarchical_cnn /data/mentali/IDS/ckpt/hierarchical_cnn_time

python main.py \
--batch_size 128 \
--model_name hierarchical_cnn \
--is_pretrain True \
--is_size True \
--train_file_path /root/data/mentali/IDS/data/cnn/train_file_compress_all.txt \
--dev_file_path /root/data/mentali/IDS/data/cnn/test_file_compress_all.txt \
--params_path /root/data/mentali/IDS/param/hierachical_cnn_params.json \

mv  /data/mentali/IDS/ckpt/hierarchical_cnn /data/mentali/IDS/ckpt/hierarchical_cnn_size
