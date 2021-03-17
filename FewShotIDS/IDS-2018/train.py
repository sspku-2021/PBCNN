from model import Siamese
import datetime
import os


def main():
    tensorboard_log_path = "./logs/siamese"
    batch_size = 32
    learning_rate = 1e-3
    number_per_classes = 10
    evaluate_each = 200
    number_of_train_iterations = 10000
    log_path = "./logs/{}_{}".format(number_per_classes, datetime.datetime.now().strftime('%m-%d-%H-%M'))
    model_name = "siamese_" + str(number_per_classes) + ".h5"

    siamese_network = Siamese(batch_size=batch_size, learning_rate=learning_rate, tensorboard_log_path=tensorboard_log_path, nb_per_classes=number_per_classes)
    validation_accuracy = siamese_network.train_siamese_network(number_of_iterations=number_of_train_iterations, support_set_size=-1,
                                                                evaluate_each=evaluate_each, log_path = log_path, model_name=model_name)
    siamese_network.model.load_weights(os.path.join("models", model_name))
    test_acc = siamese_network.data_loader.one_shot_test(siamese_network.model, -1, 200, False, open(log_path+".test.csv", "w", encoding="utf-8"))


if __name__ == "__main__":
    main()

