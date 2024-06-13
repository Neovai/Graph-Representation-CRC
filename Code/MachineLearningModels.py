import os

import torch
import torch.utils
import numpy as np
import scipy.stats
import tensorflow.keras as tk
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, Softmax
from tensorflow.keras.optimizers import RMSprop
from tensorflow.config.experimental import list_physical_devices
from tensorflow.config import set_visible_devices, get_visible_devices
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import KFold


def shuffle_list(files, labels):
    import random
    data = list(zip(files, labels))
    random.shuffle(data)
    files[:], labels[:] = zip(*data)


def load_data(data_dir):
    data = []
    label = []
    total_num = 0
    for label_type in ['Readable', 'Neutral', 'Unreadable']:
        dir_name = os.path.join(data_dir, label_type)
        file_list = os.listdir(dir_name)
        if dir_name == 'Neutral':
            file_list.sort()
            file_list = file_list[0:len(file_list) // 2]
        for f_name in file_list:
            total_num += 1
            f = open(os.path.join(dir_name, f_name), errors='ignore')
            lines = []
            if not f_name.startswith('.'):
                for line in f:
                    line = line.strip(',\n')
                    info = line.split(',')
                    info_int = []
                    count = 0
                    for item in info:
                        if count < 305:
                            info_int.append(int(item))
                            count += 1
                    while count < 305:
                        info_int.append(int(-1))
                        count += 1
                    info_int = np.asarray(info_int)
                    lines.append(info_int)

                while len(lines) < 50:
                    info_int = []
                    count = 0
                    for i in range(305):
                        if count < 305:
                            info_int.append(int(-1))
                            count += 1
                    info_int = np.asarray(info_int)
                    lines.append(info_int)
                f.close()
                lines = np.asarray(lines)
                if label_type == 'Readable':
                    label.append(2)
                elif label_type == 'Unreadable':
                    label.append(0)
                else:
                    label.append(1)
                data.append(lines)

    data = np.asarray(data)
    data = data.reshape((total_num, 50, 305, 1)) / 127
    label = np.asarray(label)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', label.shape)
    return data, label


def classifier(classification: bool):
    """
    Simple, fine-tuned CNN
    """
    model = Sequential()
    model.add(Reshape((50, 305, 1), input_shape=(50, 305)))

    model.add(Convolution2D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Convolution2D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Convolution2D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=3, strides=3))

    model.add(Flatten())
    model.add(Dense(units=128, activation='relu', kernel_regularizer=tk.regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(units=32, activation='relu'))

    if classification:
        # add final linear layer w/ softmax activation to output class predictions:
        model.add(Dense(units=3, activation='softmax'))

    rms = RMSprop(learning_rate=0.0015)

    # set optimizer, loss function, and evaluation metrics:
    model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def to_one_hot(data_labels, dimension=3):
    results = np.zeros((len(data_labels), dimension))
    for m, data_labels in enumerate(data_labels):
        results[m, data_labels] = 1
    return results


def cross_train_RF(X, Y, X_gen, Y_gen, k_folds, batch, n_epochs):
    """
    kfold cross-validation for CNN and random forest classifiers
    """
    kfold = KFold(n_splits=k_folds, shuffle=True)

    fold_test_acc = []
    fold_test_f1 = []
    fold_test_auc = []
    fold_test_mcc = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(X, Y)):
        print(f'\n ------------ FOLD {fold} ------------ \n')

        # Random Forest with CNN feature learning:
        RF_feature = classifier(classification=False)
        model = RandomForestClassifier(n_estimators=40, random_state=20, bootstrap=True,
                                             max_features='sqrt', warm_start=True)

        # concat generated data with training data (not add to test data):
        train_X = X[train_ids]
        train_Y = Y[train_ids]
        if X_gen.size > 0 and Y_gen.size > 0:
            train_X = np.concatenate((X[train_ids], X_gen[train_ids]), axis=0)
            train_Y = np.concatenate((Y[train_ids], Y_gen[train_ids]), axis=0)

        # Get features for Random Forest Classifier:
        train_X = RF_feature(train_X)
        test_X = RF_feature(X[test_ids])

        # train model on fold:
        # convert raw single Y labels to one hot vector encoding of length 3:
        # (requred to be compatible with CNN model output of softmax for each class)
        model.fit(train_X, train_Y)

        # test model on fold:
        predictions = model.predict(test_X)
        model_acc = model.score(test_X, Y[test_ids])

        # Calculate results:
        model_f1 = f1_score(Y[test_ids], predictions, average='macro')
        model_auc = roc_auc_score(np.asarray(to_one_hot(Y[test_ids])), np.asarray(to_one_hot(predictions)), average='macro', multi_class='ovo')
        model_mcc = matthews_corrcoef(Y[test_ids], predictions)

        # print fold results:
        print(f"[Fold {fold}] Test Acc: {model_acc}     | F1: {model_f1}    | AUC: {model_auc}  | MCC: {model_mcc}")

        fold_test_acc.append(model_acc)
        fold_test_f1.append(model_f1)
        fold_test_auc.append(model_auc)
        fold_test_mcc.append(model_mcc)

    return np.average(fold_test_acc), np.average(fold_test_f1), np.average(fold_test_auc), np.average(fold_test_mcc)


def cross_train(X, Y, X_gen, Y_gen, k_folds, batch, n_epochs):
    """
    kfold cross-validation for CNN
    """
    kfold = KFold(n_splits=k_folds, shuffle=True)

    fold_test_acc = []
    fold_test_f1 = []
    fold_test_auc = []
    fold_test_mcc = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(X, Y)):
        print(f'\n ------------ FOLD {fold} ------------ \n')

        # Simple CNN for results replication of paper:
        model = classifier(classification=True)

        # concat generated data with training data (not add to test data):
        train_X = X[train_ids]
        train_Y = Y[train_ids]
        if X_gen.size > 0 and Y_gen.size > 0:
            train_X = np.concatenate((X[train_ids], X_gen[train_ids]), axis=0)
            train_Y = np.concatenate((Y[train_ids], Y_gen[train_ids]), axis=0)
        print(train_ids.shape)
        print(X[train_ids].shape)
        print(train_X.shape)

        # train model on fold:
        # convert raw single Y labels to one hot vector encoding of length 3:
        # (requred to be compatible with CNN model output of softmax for each class)
        model.fit(train_X, to_one_hot(train_Y), batch_size=batch, epochs=n_epochs, verbose=2)

        # test model on fold:
        model_scores = model.evaluate(X[test_ids], to_one_hot(Y[test_ids]), verbose=0, return_dict=True)
        predictions = np.argmax(model(X[test_ids]), axis=1)

        # Calculate results:
        model_acc = model_scores["accuracy"]
        loss = model_scores["loss"]
        model_f1 = f1_score(Y[test_ids], predictions, average='macro')
        model_auc = roc_auc_score(np.asarray(to_one_hot(Y[test_ids])), np.asarray(to_one_hot(predictions)), average='macro', multi_class='ovo')
        model_mcc = matthews_corrcoef(Y[test_ids], predictions)

        # print fold results:
        print(f"[Fold {fold}] Test Acc: {model_acc}  | Loss: {loss}  | F1: {model_f1}    | AUC: {model_auc}  | MCC: {model_mcc}")

        fold_test_acc.append(model_acc)
        fold_test_f1.append(model_f1)
        fold_test_auc.append(model_auc)
        fold_test_mcc.append(model_mcc)

    return np.average(fold_test_acc), np.average(fold_test_f1), np.average(fold_test_auc), np.average(fold_test_mcc)


if __name__ == '__main__':
    # Enable CPU use only: (Don't have TensorFlow GPU support)
    cpus = list_physical_devices('CPU')
    set_visible_devices([], 'GPU')
    set_visible_devices(cpus[0], 'CPU')
    get_visible_devices()

    # Pull in original dataset:
    original_set = '../Dataset/Structure'
    X, Y = load_data(original_set)
    shuffle_list(X, Y)

    # Pull in generated dataset:
    generated_set = '../Dataset/generated_dataset'
    X_gen, Y_gen = load_data(generated_set)
    shuffle_list(X_gen, Y_gen)

    forest_acc_results = []
    forest_f1_results = []
    forest_f2_results = []
    forest_auc_results = []

    forest_acc_results1 = []
    forest_f1_results1 = []
    forest_f2_results1 = []
    forest_auc_results1 = []

    knn_acc_result = []
    knn_f1_result = []
    knn_f2_result = []
    knn_auc_result = []

    knn_acc_result1 = []
    knn_f1_result1 = []
    knn_f2_result1 = []
    knn_auc_result1 = []

    # Added kfold cross-validation:
    k_folds = 5
    batch = 15
    n_epochs = 100
    acc, f1, auc, mcc = cross_train(X, Y, X_gen, Y_gen, k_folds, batch, n_epochs)
    print(f"\n[Final Average Scores CNN] Test Acc: {acc}     | F1: {f1}    | AUC: {auc}  | MCC: {mcc}")
    acc_rf, f1_rf, auc_rf, mcc_rf = cross_train_RF(X, Y, X_gen, Y_gen, k_folds, batch, n_epochs)
    print(f"\n[Final Average Scores RF] Test Acc: {acc_rf}     | F1: {f1_rf}    | AUC: {auc_rf}  | MCC: {mcc_rf}")
