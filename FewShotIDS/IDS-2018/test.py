from model import PBCNN, Siamese
from dataloader import Dataloader


import time
import os
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def load_data(path, nb_per_classes=None):
    import json
    import random
    data = []
    with open(path, "r", encoding="utf-8") as fr:
        for idx, line in enumerate(fr):
            info = json.loads(line)
            data.append((np.array(info["pkts"]) / 255., info["label"]))
    random.shuffle(data)
    if nb_per_classes is not None:
        data = data[:nb_per_classes]
    X = [info[0] for info in data]
    y = [info[1] for info in data]
    return np.array(X), np.array(y)


def testpbcnn():
    model = PBCNN(batch_size=512, learning_rate=1e-3)
    model.construct_pbcnn()
    model.model.load_weights(os.path.join("models", "pbcnn.h5"))

    X_test, y_test = load_data(os.path.join("data", "test.json"))
    pbcnn_time = 0
    for i in range(1000):
        x_test = np.expand_dims(X_test[i], axis=0)
        start_time = time.time()
        model.model.predict_on_batch(x_test)
        end_time = time.time()
        pbcnn_time += (end_time - start_time)

    print("PBCNN: ", pbcnn_time)
    start_time = time.time()
    model.model.predict_on_batch(X_test[:1000])
    end_time = time.time()
    pbcnn_time = end_time - start_time
    print("PBCNN: ", pbcnn_time)


def test_siamese():
    batch_size = 1
    learning_rate = 1e-3
    siamese_network = Siamese(batch_size=batch_size, learning_rate=learning_rate,
                              tensorboard_log_path=None, nb_per_classes=15)
    siamese_network.model.load_weights(os.path.join("models", "siamese_100.h5"))
    siamese_network.data_loader.one_shot_test_time(siamese_network.model, -1, 67)


if __name__ == "__main__":
    testpbcnn()
    test_siamese()
