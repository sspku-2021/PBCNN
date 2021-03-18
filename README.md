Requirement

python = 3.7
tensorflow = 2.1.0
keras = 1.0

Source code in the "code" folder. preprocess.py used for preprocess original pcap files to tfrecord which fit the model input.
run.py is our model description and train, valid, test process. "config" folder contains attacker ips and label maps. "tools" folder contains tool functions.  

The process of reproducing our model:

1. Download the dataset in https://www.unb.ca/cic/datasets/ids-2018.html
2. python preprocess.py (Path in the source code need to convert to your own path)
3. python run.py (Path in the source code need to convert to your own path)

