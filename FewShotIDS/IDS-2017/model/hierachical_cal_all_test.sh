python main.py \
--batch_size 128 \
--model_name hierarchical_cnn \
--mode test \
--test_file_path /root/data/mentali/IDS/data/cnn/test_file_compress_all.txt \
--label_mapping_path /root/data/mentali/IDS/cache/label_mapping_all.pkl \
--params_path /root/data/mentali/IDS/param/hierachical_cnn_params.json \
--restore_ckpt /root/data/mentali/IDS/ckpt/hierarchical_cnn_base/hierarchical_cnn_14 \
--file_output /root/data/mentali/IDS/output/hierarchical_cnn_all.txt \


python main.py \
--batch_size 128 \
--model_name cnn_lstm \
--mode test \
--attention_modules self_attention \
--test_file_path /root/data/mentali/IDS/data/cnn/test_file_compress_all.txt \
--label_mapping_path /root/data/mentali/IDS/cache/label_mapping_all.pkl \
--params_path /root/data/mentali/IDS/param/cnn_lstm_params.json \
--restore_ckpt /root/data/mentali/IDS/ckpt/cnn_lstm/cnn_lstm_19 \
--file_output /root/data/mentali/IDS/output/hierarchical_cnn_lstm_all.txt \