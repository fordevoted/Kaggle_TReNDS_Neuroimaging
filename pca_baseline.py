
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import plotly.offline as py
# py.init_notebook_mode(connected=True)
from tensorflow.python.keras import regularizers

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv3D, Dense, Dropout, BatchNormalization, Activation
import tensorflow.keras.backend as K

from sklearn.impute import KNNImputer
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def weighted_NAE(yTrue, yPred):
    weights = K.constant([.3, .175, .175, .175, .175], dtype=tf.float32)

    return K.sum(weights * K.sum(K.abs(yTrue - yPred)) / K.sum(yPred))


train_scores = pd.read_csv("./kaggle/input/train_scores.csv")
reveal_ID_site2 = pd.read_csv("./kaggle/input/reveal_ID_site2.csv")
fnc = pd.read_csv("./kaggle/input/fnc.csv")
loading = pd.read_csv("./kaggle/input/loading.csv")
ICN_numbers = pd.read_csv("./kaggle/input/ICN_numbers.csv")
sample_submission = pd.read_csv("./kaggle/input/sample_submission.csv")

data = pd.concat([loading, fnc.drop(['Id'], axis=1)], axis=1)
train = pd.DataFrame(train_scores.Id).merge(data, on='Id')
test_id=[]
for i in range(0,len(sample_submission.Id),5):
    test_id.append(float(sample_submission.Id[i].split("_")[0]))
test = pd.DataFrame(test_id, columns=["Id"]).merge(data, on='Id')

X_train1, X_pretest1, y_train1, y_pretest1 = train_test_split(train.drop(['Id'], axis=1), pd.DataFrame(train_scores).drop('Id', axis=1), test_size=0.2, random_state=42)

impute = KNNImputer(n_neighbors=40)
y_train2 = impute.fit_transform(y_train1)
y_pretest2 = impute.transform(y_pretest1)

scaler = StandardScaler()
train2 = scaler.fit_transform(X_train1)
pretest2 = scaler.transform(X_pretest1)
test2 = scaler.transform( test.drop(['Id'], axis=1))
print(train2.shape)
pca = KernelPCA(n_components=450, kernel='cosine')# more than 0.95
train3 = pca.fit_transform(train2)
print(train3.shape)
pretest3 = pca.transform(pretest2)
test3 = pca.transform(test2)
input_dim = train3.shape[1]
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

model.add(Dense(512, kernel_initializer="lecun_normal",
                kernel_regularizer=regularizers.l2(regularize_rate)))
# model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(Dropout(0.4))

model.add(Dense(256, kernel_initializer="lecun_normal",
                kernel_regularizer=regularizers.l2(regularize_rate)))
# model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(Dropout(0.2))

# model.add(Dense(128, kernel_initializer="lecun_normal",
#                 kernel_regularizer=regularizers.l2(regularize_rate)))
# model.add(BatchNormalization())
# model.add(Activation('selu'))
# model.add(Dropout(0.4))
model.add(Dense(5, activation='relu'))

model = model
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.0001), metrics=[ weighted_NAE])
model.summary()

filepath = "./best.hdf5"

checkpoint = ModelCheckpoint(filepath,
                            monitor='val_weighted_NAE',
                            verbose=1,
                            save_best_only=True,
                            mode='min',
                             save_weights_only=True)
history = model.fit(train3, y_train2, validation_data=(pretest3, y_pretest2), epochs=30, batch_size=16, callbacks=[checkpoint])
model.load_weights(filepath)

pred=pd.DataFrame()
pred["Id"]=sample_submission.Id
pred["Predicted"]=model.predict(test3).flatten()
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