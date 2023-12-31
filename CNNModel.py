import matplotlib
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings;

from spacy import vocab
from tensorflow.python.keras.initializers.initializers_v2 import Constant

warnings.filterwarnings(action='once')
# import seaborn as sns
# from waffle import Waffle

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.utils import to_categorical
import keras


# Keras version.
print('Using Keras version', keras.__version__)

# sns.set(style="darkgrid")

RANDOM_SEED = 42

# Read data.
data_path = 'mushrooms.csv'
df = pd.read_csv(data_path)
df.head()

df.info()


# Print unique values for columns.
columns = df.columns
for col in columns:
    print('{feat_name}: {feat_values}'.format(feat_name=col, feat_values=df[col].unique()))

# First, create the list of labels as our y values.
le = preprocessing.LabelEncoder()
y = le.fit_transform(df['class'])
print(y)

# Drop the labels from the dataframe, encode all features.
X = df.drop('class', axis=1)
columns = X.columns
for i in range(len(X.columns)):
    le = preprocessing.LabelEncoder()
    X[columns[i]] = le.fit_transform(X[columns[i]])
    
X.head()

# Inspect unique values again.
for col in columns:
    print('{}: {}'.format(col, X[col].unique()))


# We need to know the maximum number of possible values for the embedding layer.
# If we were using text, this would be the size of the vocabulary.
# Find number of uniabsque values for each feature.
for col in columns:
    print('{}: {}'.format(col, X[col].nunique()))


# Convert features to sequences.
sequences = []
# seq = '{}' * 22
columns = X.columns
for idx, row in X.iterrows():
    sequence = []
    for i in range(len(columns)):
        sequence.append(row[columns[i]])
    sequences.append(sequence)
    
# Print first example and label, length of example sequence.
print('{sequence}: {label}'.format(sequence=sequences[0], label=y[0]))
print('len of sequences:', len(sequences[0]))



# Build train/test sets.
x_train, x_test, y_train, y_test = train_test_split(sequences, y,
                                                    test_size=0.1,
                                                    random_state=RANDOM_SEED)
# Convert to numpy arrays.
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(x_train)
print(y_train)


def build_model():
    embeddings_dims = 300
    max_seq_length = len(sequences[0])
    max_features = 12
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    
    using_pretrained_emb = False #@param {type:"boolean"}

    # CNN via Keras.
    model = Sequential()

    if using_pretrained_emb:
      model.add(layers.Embedding(max_features,
                                 embeddings_dims,
                                 embeddings_initializer=Constant(vocab),
                                 input_length=max_seq_length,
                                 trainable=False))
    else:
      model.add(layers.Embedding(max_features,
                                 embeddings_dims,
                                 input_length=max_seq_length))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv1D(filters,
                            kernel_size,
                            padding='valid',
                            activation='relu',
                            strides=1))
    model.add(layers.MaxPooling1D())
    model.add(layers.Conv1D(filters,
                            kernel_size,
                            padding='valid',
                            activation='relu',
                            strides=1))
    model.add(layers.MaxPooling1D())
    model.add(layers.Conv1D(filters,
                            kernel_size,
                            padding='valid',
                            activation='relu',
                            strides=1))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(hidden_dims))
    model.add(layers.Dropout(0.5))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    
    return model

# Build the model, check out the summary.
model = build_model()
model.summary()

# Train.
history = model.fit(x_train, y_train,
                    epochs=10,
                    verbose=True,
                    validation_data=(x_test, y_test),
                    batch_size=1)


# Get predictions for the test set.
preds_test_prob = model.predict(x_test)
preds_test = (preds_test_prob > 0.5).astype(int)

# Print the predicted probabilities and predicted class for the first 10 samples.
for i in range(10):
    print(f"Sample {i + 1}: Predicted Probabilities = {preds_test_prob[i]}, Predicted Class = {preds_test[i]}, True Class = {y_test[i]}")


cnn_metrics = {'acc': metrics.accuracy_score(y_test, preds_test)}
cnn_metrics['prec'] = metrics.precision_score(y_test, preds_test)
cnn_metrics['rec'] = metrics.recall_score(y_test, preds_test)
cnn_metrics['f1'] = metrics.f1_score(y_test, preds_test)
cnn_metrics['f1_macro'] = metrics.f1_score(y_test, preds_test,
                                           average='macro')
cnn_metrics['auc'] = metrics.roc_auc_score(y_test, preds_test)

for metric in cnn_metrics:
  print('{metric_name}: {metric_value}'.format(metric_name=metric, metric_value=cnn_metrics[metric]))

# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Get training and test accuracy history.
# training_acc = history.history['acc']
# test_acc = history.history['val_acc']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Visualize acc history
# plt.plot(epoch_count, training_acc, 'r--')
# plt.plot(epoch_count, test_acc, 'b-')
# plt.legend(['Training Acc', 'Test Acc'])
# plt.xlabel('Epoch')
# plt.ylabel('Acc')
# plt.show()
