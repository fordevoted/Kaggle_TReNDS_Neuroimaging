# Libraries
import gc
import random
import warnings

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Activation, BatchNormalization, regularizers, Convolution1D
from keras.layers import Dropout
from keras.models import Sequential
from numpy.random import seed
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import keras.backend as K
import keras.optimizers as opt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
filepath = "weights_best.hdf5"


def l4_norm_distance(y_true, y_pred):
    # y_true = tf.constant([[1, 2.0, 3.0, 1.2, 4.1]])
    # y_pred = tf.constant([[2, 2.1, 1.2, 4.1, 1.3]])
    return K.pow(K.mean(K.pow(y_pred - y_true, 4), axis=-1), 1/4)
    # return K.pow(K.mean(K.pow(y_true - y_pred, 1), axis=-1), 1)

def weighted_NAE(yTrue, yPred):
    weights = K.constant([.3, .175, .175, .175, .175], dtype=tf.float32)

    return K.sum(weights * K.sum(K.abs(yTrue - yPred)) / K.sum(yPred))

def outlier_2s(df):
    for i in range(1, len(df.columns) - 1):
        col = df.iloc[:, i]
        average = np.mean(col)
        sd = np.std(col)
        outlier_min = average - (sd) * 2.2
        outlier_max = average + (sd) * 2.2
        col[col < outlier_min] = outlier_min
        col[col > outlier_max] = outlier_max
    return df


def scaler(df):
    for i in range(5, len(df.columns) - 5):
        col = df.iloc[:, i]
        col = preprocessing.minmax_scale(col)
    return df


def mean_diff1(df):
    for i in range(7, 7 + len(loading.columns)):
        dfa = df.iloc[:, 7:7 + len(loading.columns)]
        average = np.mean(np.mean(dfa))
        col = df.iloc[:, i]
        for j in range(1, len(train)):
            val = df.iloc[j]
            val = col - average
    return df


def mean_diff2(df):
    for i in range(7 + len(loading.columns), 7 + len(loading.columns) + len(fnc.columns) - 7):
        dfa = df.iloc[:, 7 + len(loading.columns):7 + len(loading.columns) + len(fnc.columns)]
        average = np.mean(np.mean(dfa))
        col = df.iloc[:, i]
        for j in range(1, len(train)):
            val = df.iloc[j]
            val = col - average
    return df


workspace = './kaggle/input/'

if __name__ == "__main__":
    seed(42)
    random.seed(42)

    # train = pd.read_csv(workspace + 'train_scores.csv', dtype={'Id': str}) \
    #     .dropna().reset_index(drop=True)  # to make things easy
    train = pd.read_csv(workspace + 'train_scores.csv', dtype={'Id': str})
    for col in train.columns:
        print(col)
        if col != 'Id':
            train[col] = train[col].fillna(train[col].mean())

    reveal_ID = pd.read_csv(workspace + 'reveal_ID_site2.csv', dtype={'Id': str})
    ICN_numbers = pd.read_csv(workspace + 'ICN_numbers.csv')
    loading = pd.read_csv(workspace + 'loading.csv', dtype={'Id': str})
    fnc = pd.read_csv(workspace + 'fnc.csv', dtype={'Id': str})
    sample_submission = pd.read_csv(workspace + 'sample_submission.csv', dtype={'Id': str})

    # Config
    OUTPUT_DICT = ''
    ID = 'Id'
    TARGET_COLS = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
    SEED = 42

    sample_submission['ID_num'] = sample_submission[ID].apply(lambda x: int(x.split('_')[0]))
    test = pd.DataFrame({ID: sample_submission['ID_num'].unique().astype(str)})
    del sample_submission['ID_num']
    gc.collect()

    # merge
    train = train.merge(loading, on=ID, how='left')
    train = train.merge(fnc, on=ID, how='left')

    test = test.merge(loading, on=ID, how='left')
    test = test.merge(fnc, on=ID, how='left')

    len(loading.columns)

    len(fnc.columns)

    len(train.columns)

    train = outlier_2s(train)
    train = scaler(train)
    train = train.dropna(how='all').dropna(how='all', axis=1)

    X_train = train.drop('Id', axis=1).drop(TARGET_COLS, axis=1)
    y_train = train.drop('Id', axis=1)[TARGET_COLS]
    X_test = test.drop('Id', axis=1)

    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    #
    # pca = PCA(n_components=445)  # more than 0.95
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)

    np.random.seed(1964)
    epochs = 200
    batch_size = 32
    verbose = 2
    validation_split = 0.2
    input_dim = X_train.shape[1]
    n_out = y_train.shape[1]
    regularize_rate = 5.0e-4

    model = Sequential()
        # input
    # model.add(Convolution1D(nb_filter=512, filter_length=1, input_shape=(input_dim, )))
    model.add(BatchNormalization(input_shape=(input_dim, )))
    model.add(Dense(512, kernel_initializer="lecun_normal", use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('selu'))
    # model.add(Dropout(0.2))

    model.add(Dense(1024, kernel_initializer="lecun_normal", use_bias=False, kernel_regularizer=regularizers.l2(regularize_rate)))
    model.add(BatchNormalization())
    model.add(Activation('selu'))
    # model.add(Dropout(0.2))

    model.add(Dense(2048, kernel_initializer="lecun_normal", use_bias=False, kernel_regularizer=regularizers.l2(regularize_rate)))
    model.add(BatchNormalization())
    model.add(Activation('selu'))
    # model.add(Dropout(0.4))

    model.add(Dense(256, kernel_initializer="lecun_normal", use_bias=False, kernel_regularizer=regularizers.l2(regularize_rate)))
    model.add(BatchNormalization())
    model.add(Activation('selu'))
    # model.add(Dropout(0.2))
    model.add(Dense(n_out, activation='relu'))

    model.compile(loss=weighted_NAE,
                    optimizer=opt.Adam(lr=1.0e-3),
                    metrics=['mean_squared_error', weighted_NAE])

    early_stopping = EarlyStopping(monitor='val_weighted_NAE', patience=10, verbose=2)
    checkPoint = ModelCheckpoint(filepath, monitor='val_weighted_NAE', verbose=2, save_best_only=True,
                                 mode='min')

    history = model.fit(X_train, y_train,
                         batch_size=batch_size, epochs=epochs,
                         callbacks=[early_stopping, checkPoint],
                         verbose=verbose, validation_split=validation_split)
    model.load_weights("weights_best.hdf5")
    prediction_dict = model.predict(X_test)
    prediction_dict = pd.DataFrame(prediction_dict)
    prediction_dict.columns = y_train.columns
    prediction_dict.head(10)

    pred_df = pd.DataFrame()

    for TARGET in TARGET_COLS:
        tmp = pd.DataFrame()
        tmp[ID] = [f'{c}_{TARGET}' for c in test[ID].values]
        tmp['Predicted'] = prediction_dict[TARGET]
        pred_df = pd.concat([pred_df, tmp])

    print(pred_df.shape)
    print(sample_submission.shape)

    pred_df.head()

    # submission
    submission = pd.merge(sample_submission, pred_df, on='Id')[['Id', 'Predicted_y']]
    submission.columns = ['Id', 'Predicted']

    submission.to_csv('submission.csv', index=False)
    submission.head()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
