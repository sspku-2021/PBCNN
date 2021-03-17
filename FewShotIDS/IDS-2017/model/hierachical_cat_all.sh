python3 main.py \
--batch_size 128 \
--model_name hierarchical_cnn \
--train_file_path /root/data/mentali/IDS/data/cnn/train_file_compress_all.txt \
--dev_file_path /root/data/mentali/IDS/data/cnn/test_file_compress_all.txt \
--label_mapping_path /root/data/mentali/IDS/cache/label_mapping_all.pkl \
--params_path /root/data/mentali/IDS/param/hierachical_cnn_params.json \

python main.py \
--batch_size 128 \
--model_name cnn_lstm \
--attention_modules self_attention \
--num_epochs 20 \
--is_early_stop False \
--train_file_path /root/data/mentali/IDS/data/cnn/train_file_compress_all.txt \
--dev_file_path /root/data/mentali/IDS/data/cnn/test_file_compress_all.txt \
--label_mapping_path /root/data/mentali/IDS/cache/label_mapping_all.pkl \
--params_path /root/data/mentali/IDS/param/cnn_lstm_params.json \


