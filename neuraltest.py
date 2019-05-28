from sklearn import datasets, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from keras import Sequential
from sklearn import metrics
from keras.layers import Dense, Input, MaxPooling1D, Conv1D, GlobalAveragePooling1D, MaxPooling2D, Conv2D, GlobalMaxPooling1D, SpatialDropout1D, Dropout, Flatten, Embedding, merge, InputSpec, Layer
import keras.layers.merge
from keras.layers.merge import concatenate
import keras
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import tensorflow as tf
import os
import sys
from keras.optimizers import SGD
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping
from keras import optimizers
from keras import  Model
from keras.preprocessing import sequence
class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """

    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]

        # return flattened output
        return Flatten()(top_k)
def get_kmax_text_cnn(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size):
    embedding_layer = Embedding(nb_words,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=False)

    filter_nums = 180
    drop = 0.6

    comment_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(comment_input)
    embedded_sequences = SpatialDropout1D(0.2)(embedded_sequences)

    conv_0 = Conv1D(filter_nums, 1, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_1 = Conv1D(filter_nums, 2, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_2 = Conv1D(filter_nums, 3, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_3 = Conv1D(filter_nums, 4, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)

    # conv_0 = Conv1D(filter_nums / 2, 1, kernel_initializer="normal", padding="valid", activation="relu")(conv_0)
    # conv_1 = Conv1D(filter_nums / 2, 2, kernel_initializer="normal", padding="valid", activation="relu")(conv_1)
    # conv_2 = Conv1D(filter_nums / 2, 3, strides=2, kernel_initializer="normal", padding="valid", activation="relu")(conv_2)

    maxpool_0 = KMaxPooling(k=3)(conv_0)
    maxpool_1 = KMaxPooling(k=3)(conv_1)
    maxpool_2 = KMaxPooling(k=3)(conv_2)
    maxpool_3 = KMaxPooling(k=3)(conv_3)

    #merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2, maxpool_3], mode='concat', concat_axis=1)
    merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2, maxpool_3], axis=1)
    output = Dropout(drop)(merged_tensor)
    output = Dense(units=144, activation='relu')(output)
    output = Dense(units=out_size, activation='sigmoid')(output)

    model = Model(inputs=comment_input, outputs=output)
    adam_optimizer = optimizers.Adam(lr=1e-3, decay=1e-7)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    model.summary()
    return model
def calculate_false_positive(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for j in range(0, y_true.shape[1]):
        for i in range(0, y_true.shape[1]):
            if y_pred[j][i] == y_true[j][i]:
                if y_pred[j][i] == 0:
                    tn += 1
                else:
                    tp += 1
            else:
                if y_pred[j][i] == 0:
                    fn += 1
                else:
                    fp += 1
    print("True positive", tp)
    print("False positive", fp)
    print("True negative", tn)
    print("False negative", fn)
def change_list(l):
    l = np.array(l)
    STARTDIF = 0.2
    CHANGE = 0.6
    for i in range(0, l.shape[0]):
        for j in range(0, l.shape[1]):
            l[i][j] = l[i][j] * CHANGE + STARTDIF
    return l
#import keras_metrics

def get_text(file):
    filename = "D:\\TestFlibusta\\" + file
    text_number = filename.split('.')[0]
    encodings = [None, 'utf-8', 'ISO-8859-1']
    if file.endswith(".fb2") and os.path.isfile(text_number+"Tags.txt"):
        for encoding in encodings:
            try:
                f = open(filename, 'r', encoding=encoding)
                s = f.read()
                f.close()
                return s
            except UnicodeDecodeError:
                s = ""
                #print("UnicodeDecode")
    print("Very bad error")
    return ""
# define roc_callback, inspired by https://github.com/keras-team/keras/issues/6050#issuecomment-329996505
def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
def get_conv_model1(features_dim, labels_dim):
    model = Sequential()
    model.add(Embedding(20000,
                        50,
                        input_length=features_dim))
    model.add(Dropout(0.2))
    model.add(Conv1D(100, 10, activation='relu'))
    model.add(Conv1D(100, 10, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(160, 10, activation='relu'))
    model.add(Dense(labels_dim, activation='relu'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                 optimizer=sgd,
                  metrics=['accuracy'])
    return model
def get_conv_model2(features_dim, labels_dim):
    model = Sequential()
    filters = 250
    kernel_size = 3
    model.add(Embedding(3000,
                        50,
                        input_length=features_dim))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1, input_shape=(713, features_dim)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(250, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(labels_dim, activation='relu'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                 optimizer=sgd,
                  metrics=['accuracy'])
    return model
def get_conv_model(features_dim, labels_dim):
    model = Sequential()
    filter_nums = 180
    kernel_size = 3
    drop = 0.6
    comment_input = Input(shape=(features_dim, ), dtype='int32')
    embedded_sequences = Embedding(3000,
                        50)(comment_input)
    embedded_sequences = Dropout(0.2)(embedded_sequences)
    model.add(SpatialDropout1D(0.2))

    conv_0 = Conv1D(filter_nums, 1, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_1 = Conv1D(filter_nums, 2, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_2 = Conv1D(filter_nums, 3, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_3 = Conv1D(filter_nums, 4, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)

    # conv_0 = Conv1D(filter_nums / 2, 1, kernel_initializer="normal", padding="valid", activation="relu")(conv_0)
    # conv_1 = Conv1D(filter_nums / 2, 2, kernel_initializer="normal", padding="valid", activation="relu")(conv_1)
    # conv_2 = Conv1D(filter_nums / 2, 3, strides=2, kernel_initializer="normal", padding="valid", activation="relu")(conv_2)

    maxpool_0 = KMaxPooling(k=3)(conv_0)
    maxpool_1 = KMaxPooling(k=3)(conv_1)
    maxpool_2 = KMaxPooling(k=3)(conv_2)
    maxpool_3 = KMaxPooling(k=3)(conv_3)

    merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2, maxpool_3], mode='concat', concat_axis=1)
    output = Dropout(drop)(merged_tensor)
    output = Dense(units=144, activation='relu')(output)
    output = Dense(units=labels_dim, activation='sigmoid')(output)

    model = Model(inputs=comment_input, outputs=output)
    adam_optimizer = optimizers.Adam(lr=1e-3, decay=1e-7)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    model.summary()
    return model
#fake_x = np.random.rand(100, 20000)
#fake_y = np.ones((100, 7))
#model = get_conv_model(fake_x.shape[1], fake_y.shape[1])
#model.fit(fake_x, fake_y, epochs = 10)
#0/0
#X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])  # create an array
#y = np.array([1, 2, 3, 4])  # Create another array
#kf = KFold(n_splits=4, shuffle=True)  # Define the split - into 2 folds
#kf.get_n_splits(X)  # retfor train_index, test_index in kf.split(X):
#for train_index, test_index in kf.split(X):
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]
#    print(X_train)
#print(kf)
#print(change_list([[0, 1], [0.5, 0.5]]))
#calculate_false_positive([[0, 1], [1, 0]], [[0, 0], [1, 1]])
#0/0
y_neo = np.array([[0,0,0], [0,1,0]])
y_pred = np.array([[1,0,0], [0,1,0]])
#print(metrics.roc_auc_score(y_neo, y_pred))
#print(metrics.log_loss(y_neo, y_pred))
#print(metrics.hamming_loss(y_neo, y_pred))
print(metrics.f1_score(y_neo, y_pred, average='macro'))
#0/0
directory = "D:\\TestFlibusta\\VECT"
d = {}
l = []
names = []
i = 0
print("IWH1")
for file in os.listdir(directory):
    if file.endswith(".txt") and file[:4] == 'VECT':
        fb2_name = file[4:].split('.')[0] + ".fb2"
        fb2_text = get_text(fb2_name)
        if len(fb2_text) > 0:
            l.append(fb2_text)
            filename = os.path.join(directory, file)
            f = open(filename, "r")
            s = f.readline()
            names.append(file)
            d[i] = [int(c) for c in s.split() if c.isdigit()]
            i += 1
            #print(d[file])
            f.close()
            if i > 2000:
                break
print("IWH2")
NUM_THRESHOLD = 30
d1 = {}
for key in list(d.keys()):
    d1[key] = []
for tagi in range(len(d[0])):
    num_of_keys = 0
    for key in list(d.keys()):
        if(d[key][tagi] == 1):
            num_of_keys += 1
    if num_of_keys > NUM_THRESHOLD:
        for key in list(d.keys()):
            d1[key].append(d[key][tagi])
d = d1
print("Number of useful tags:")
print(len(d1[0]))
print(sum(d1[0]))
#sys.exit()
#0/0
print(len(l))
print(len(list(d.keys())))
stopWords = stopwords.words('russian')
vectorizer = TfidfVectorizer(analyzer = "word" , tokenizer = None , preprocessor = None , \
stop_words = stopWords , max_features = 20000)
print("TFVECTORISED")
textlist_tfidf = vectorizer.fit_transform(l).toarray()
X = np.array(textlist_tfidf)
y = np.array(list(d.values()))
n = len(list(d.values()))
#lm = linear_model.LinearRegression()
#y = data.values()
#y = [1.0, 0.0, 0.0]
#neodata = [ ['фантастика'], ['фентези'], ['детектив']]
#X, y = [[1.0, 0.0], [0.0, 2.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [0.0, 3.0], [0.0, 4.0], [0.0, 5.0]], [1, 0, 1, 1, 1, 0, 0, 0]
print(X.shape)
print(y.shape)

#X_train, X_test, y_train, y_test = train_test_split(X, y)
#print(y_train)
#model = lm.fit(X_train, y_train)
#model = lm.fit(X, y)
#predictions = lm.predict([[0.1, 0.73]])
#print(predictions)
#classifier = Sequential()
#First Hidden Layer
#classifier.add(Dense(3, activation='relu', kernel_initializer='random_normal', input_dim=2))

#Second  Hidden Layer
#classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))

#Output Layer
#classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
#classifier.add(Dense(1, activation='sigmoid'))
#classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
#Fitting the data to the training dataset
#classifier.fit(X,y, epochs=1, batch_size=3)
#is_tag = 0

#model = Sequential()
#model.add(Dense(1024, activation='relu', input_dim=X.shape[1]))
#model.add(Dense(1024, activation='sigmoid', input_dim=X.shape[1]))
#model.add(Dense(1024, activation='sigmoid', input_dim=X.shape[1]))
#model.add(Dense(1024, activation='relu', input_dim=X.shape[1]))
# model_m.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
#model.add(Input(shape=(X.shape[1],)))
#model.add(Conv1D(100, 10, activation='relu', input_shape=(X.shape[1],)))
#model.add(MaxPooling1D(3))
#model.add(Conv1D(160, 10, activation='relu'))
#model.add(Conv1D(160, 10, activation='relu'))
#model.add(GlobalAveragePooling1D())
#model.add(Dense(y.shape[1], activation='relu'))
#model.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
#model.add(Conv1D(100, 10, activation='relu'))
#model.add(MaxPooling1D(3))
#model.add(Conv1D(160, 10, activation='relu'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=6)
#model.compile(loss='binary_crossentropy',
 #             optimizer='adam',
              #metrics=['accuracy', auc_roc])
  #           metrics=['accuracy'])
#model = get_kmax_text_cnn(100000, )
model = get_conv_model(X.shape[1], y.shape[1])
#model.compile(loss='binary_crossentropy',
#             optimizer=sgd,
#              metrics=['accuracy'])
            #metrics=['accuracy', auc_roc, auc])
             # metrics=[auc_roc])
            #metrics=[keras_metrics.precision(), keras_metrics.recall()])
              #metrics = [auc])
#my_callbacks = [EarlyStopping(monitor='auc_roc', patience=300, verbose=1, mode='max')]

kf = KFold(n_splits=2)  # Define the split - into 2 folds
kf.get_n_splits(X)  # retfor train_index, test_index in kf.split(X):
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(X_train.shape)
    print(X_test.shape)
    true_x = np.array(X_train)
    true_y = np.array(y_train)


    #model.compile(optimizer='rmsprop',
     #             loss='categorical_crossentropy',
      #            metrics=['accuracy'])

    #data = np.random.random((6, 2))

    #x_test = np.random.random((1, 2))
    #y_test = keras.utils.to_categorical(np.random.randint(2, size=(1, 1)), num_classes=2)
    # Convert labels to categorical one-hot encoding
    #one_hot_labels = keras.utils.to_categorical(true_y, num_classes=n)
    one_hot_labels = true_y
    print(one_hot_labels.shape)

    #my_callbacks = [EarlyStopping(monitor='auc_roc', patience=300, verbose=1, mode='max')]
    # Train the model, iterating on the data in batches of 32 samples
    #model.fit(true_x, one_hot_labels, epochs=100, batch_size=64, callbacks=my_callbacks)
    #model.fit(true_x, one_hot_labels, epochs=100, batch_size=64,
     #         callbacks=[early_stopping])
    true_train = sequence.pad_sequences(true_x, maxlen = X.shape[1])
    model.fit(true_train, one_hot_labels, epochs=1, batch_size=16)
    print(true_y)
    #model.fit(true_x, one_hot_labels, epochs=10, batch_size=32)
    x_test = np.array(X_test)
    y_test = np.array(y_test)
    very_true_test = sequence.pad_sequences(y_test, maxlen=X.shape[1])
    #y_test = keras.utils.to_categorical(y_test, num_classes=n)
    #print(y_test)
    #print(x_test.shape)
    #score = model.evaluate(x_test, y_test, batch_size=64)
    score = model.evaluate(very_true_test, y_test, batch_size=16)
    #print(score)
    #check_data = [[0.0, 2.0]]
    #check_data = np.array(check_data)
    print(score)
    print('Accuracy:' + str(score[1]))
    print(true_x.shape)
    print(x_test.shape)
    #prediction = model.predict(x_test)
    #fpr, tpr, thresholds = metrics.roc_auc_score(y_test, prediction)
    #print(y_test.values.argmax(axis=1))
    #print(fpr)
    #print(tpr)
    #print(thresholds)
    prediction = model.predict(very_true_test, batch_size=16)
    countnotnull = 0
    countone = 0
    for i in prediction:
        for j in i:
            if j > 0.1:
                countnotnull += 1
            if j > 0.5:
                countone += 1
    print(countnotnull)
    print("Count one", countone)
    print("Logistic loss:")
    print(metrics.log_loss(y_test, prediction))
    print(metrics.log_loss(y_test, np.zeros(y_test.shape)))
    neo_prediction = model.predict(true_x, batch_size=16)
    print(metrics.log_loss(true_y, neo_prediction))
    new_pred = (prediction > 0.5)
    calculate_false_positive(y_test, np.array(new_pred))
    print(new_pred.shape)
    print("Hamming loss:")
    print(metrics.hamming_loss(y_test, new_pred))
    print(metrics.hamming_loss(y_test, np.zeros(y_test.shape)))
    print("F1 Score:")
    print(metrics.f1_score(y_test, new_pred, average='micro'))
    print(metrics.f1_score(y_test, np.zeros(y_test.shape), average='micro'))
    print("Jaccard score:")
    print(metrics.jaccard_similarity_score(y_test, new_pred))
    print(metrics.jaccard_similarity_score(y_test, np.zeros(y_test.shape)))
    neo_change = change_list(y_test)
    pred_change = change_list(prediction)
    print("Roc Auc:")
    print(metrics.roc_auc_score(neo_change, pred_change))
    #print(metrics.roc_auc_score(y_test, new_pred))
    break
#prediction = model.predict
#print(model.predict(check_data))