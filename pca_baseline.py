import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# import plotly.offline as py
# py.init_notebook_mode(connected=True)
from scipy.special import boxcox1p
from scipy.stats import boxcox_llf
from tensorflow.python.keras import regularizers

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv3D, Dense, Dropout, BatchNormalization, Activation
import tensorflow.keras.backend as K

from scipy.stats import skew
from sklearn.impute import KNNImputer
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def weighted_NAE(yTrue, yPred):
    weights = K.constant([.3, .175, .175, .175, .175], dtype=tf.float32)

    return K.sum(weights * K.sum(K.abs(yTrue - yPred)) / K.sum(yPred))


# load data
train_scores = pd.read_csv("./kaggle/input/train_scores.csv")
reveal_ID_site2 = pd.read_csv("./kaggle/input/reveal_ID_site2.csv")
fnc = pd.read_csv("./kaggle/input/fnc.csv")
loading = pd.read_csv("./kaggle/input/loading.csv")
ICN_numbers = pd.read_csv("./kaggle/input/ICN_numbers.csv")
sample_submission = pd.read_csv("./kaggle/input/sample_submission.csv")
# prepare data
data = pd.concat([loading, fnc.drop(['Id'], axis=1)], axis=1)
train = pd.DataFrame(train_scores.Id).merge(data, on='Id')
test_id = []
for i in range(0, len(sample_submission.Id), 5):
    test_id.append(float(sample_submission.Id[i].split("_")[0]))
test = pd.DataFrame(test_id, columns=["Id"]).merge(data, on='Id')

train = train.drop(['Id'], axis=1)
test = test.drop(['Id'], axis=1)
ntrain = train.shape[0]
print(train.shape, test.shape)
label = pd.DataFrame(train_scores).drop('Id', axis=1)

# Complementary label
impute = KNNImputer(n_neighbors=40)
label = impute.fit_transform(label)
print("label shape", label.shape)

# increase kurtosis
# label = np.log1p(label)

# concat to all data
all_data = pd.concat([train, test], axis=0)
print("after concat", all_data.shape)


# # build more feature
# all_data['append_feature1'] = all_data['IC_17'] * all_data['IC_04']
# all_data['append_feature2'] = all_data['IC_14'] * all_data['IC_13']
# all_data['append_feature3'] = all_data['IC_14'] - all_data['IC_22']
# all_data['append_feature4'] = all_data['IC_22'] - all_data['IC_10']
# all_data['append_feature5'] = all_data['IC_02'] * all_data['IC_15'] * all_data['IC_11']

# increase kurtosis of training data
# all_data = np.log1p(all_data)

# skew data correctness  and Check the skew of all numerical features
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew': skewed_feats})
skewness = skewness[abs(skewness['Skew']) > 0.5]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

skewed_features = skewness.index
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], 0.5)

# standardize
scaler = StandardScaler()
all_data = scaler.fit_transform(all_data)
print("after scalar", all_data.shape)

# dimension reduction
pca = KernelPCA(n_components=450, kernel='cosine')  # more than 0.95
all_data = pca.fit_transform(all_data)
print("after pca", all_data.shape)

# spilt dataset
train = all_data[:ntrain]
test = all_data[ntrain:]
print("after spilt", train.shape, test.shape)
X_train, X_validation, y_train, y_validation = train_test_split(train, label, test_size=0.2, random_state=42)

input_dim = X_train.shape[1]
regularize_rate = 5.0e-5
# model = Sequential()
model = Sequential()
# input
# model.add(Convolution1D(nb_filter=512, filter_length=1, input_shape=(input_dim, )))
# model.add(BatchNormalization())
model.add(Dense(512, kernel_initializer="lecun_normal", input_shape=(input_dim,)))
# model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(Dropout(0.4))
model.add(Dense(256, kernel_initializer="lecun_normal",
                kernel_regularizer=regularizers.l2(regularize_rate)))
# model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(Dropout(0.2))

model.add(Dense(512, kernel_initializer="lecun_normal",
                kernel_regularizer=regularizers.l2(regularize_rate)))
# model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(Dropout(0.4))

# model.add(Dense(128, kernel_initializer="lecun_normal",
#                 kernel_regularizer=regularizers.l2(regularize_rate)))
# model.add(BatchNormalization())
# model.add(Activation('selu'))
# model.add(Dropout(0.4))
model.add(Dense(5, activation='relu'))

model = model
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.0001), metrics=[weighted_NAE])
model.summary()

filepath = "./best.hdf5"

checkpoint = ModelCheckpoint(filepath,
                             monitor='val_weighted_NAE',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             save_weights_only=True)
history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=30, batch_size=8,
                    callbacks=[checkpoint])
model.load_weights(filepath)

pred = pd.DataFrame()
pred["Id"] = sample_submission.Id

# pred["Predicted"] = np.expm1(model.predict(test).flatten())
pred["Predicted"] = model.predict(test).flatten()
pred.to_csv('out2.csv', index=False)

# plt.plot(history.history['weighted_NAE'])
plt.plot(history.history['val_weighted_NAE'])
plt.title('model weighted_NAE')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['test'], loc='upper left')
plt.show()

'''
0.1658
model.add(BatchNormalization(input_shape=(input_dim,)))
model.add(Dense(512, kernel_initializer="lecun_normal", use_bias=False))
model.add(BatchNormalization())
model.add(Activation('selu'))
# model.add(Dropout(0.2))

model.add(Dense(1024, kernel_initializer="lecun_normal", use_bias=False,
                kernel_regularizer=regularizers.l2(regularize_rate)))
model.add(BatchNormalization())
model.add(Activation('selu'))
# model.add(Dropout(0.2))

model.add(Dense(2048, kernel_initializer="lecun_normal", use_bias=False,
                kernel_regularizer=regularizers.l2(regularize_rate)))
model.add(BatchNormalization())
model.add(Activation('selu'))
# model.add(Dropout(0.4))

model.add(Dense(256, kernel_initializer="lecun_normal", use_bias=False,
                kernel_regularizer=regularizers.l2(regularize_rate)))
model.add(BatchNormalization())
model.add(Activation('selu'))
# model.add(Dropout(0.2))
model.add(Dense(5, activation='relu'))

'''
