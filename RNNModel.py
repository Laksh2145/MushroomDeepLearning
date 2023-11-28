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

# Label Encoding
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
# Find number of unique values for each feature.
for col in columns:
    print('{}: {}'.format(col, X[col].nunique()))

# Convert features to sequences.
sequences = []
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


def build_rnn_model():
    embeddings_dims = 300
    max_seq_length = len(sequences[0])
    max_features = 12
    hidden_dims = 250

    using_pretrained_emb = False  # Set to True if using pretrained embeddings

    # RNN via Keras.
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
    model.add(layers.SimpleRNN(hidden_dims, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model

# Build the RNN model, check out the summary.
rnn_model = build_rnn_model()
rnn_model.summary()

# Train RNN.
rnn_history = rnn_model.fit(x_train, y_train,
                            epochs=10,
                            verbose=True,
                            validation_data=(x_test, y_test),
                            batch_size=1)

# Get predictions for the test set.
rnn_preds_test_prob = rnn_model.predict(x_test)
rnn_preds_test = (rnn_preds_test_prob > 0.5).astype(int)

# Print the predicted probabilities and predicted class for the first 10 samples.
for i in range(10):
    print(f"Sample {i + 1}: Predicted Probabilities = {rnn_preds_test_prob[i]}, Predicted Class = {rnn_preds_test[i]}, True Class = {y_test[i]}")

rnn_metrics = {'acc': metrics.accuracy_score(y_test, rnn_preds_test)}
rnn_metrics['prec'] = metrics.precision_score(y_test, rnn_preds_test)
rnn_metrics['rec'] = metrics.recall_score(y_test, rnn_preds_test)
rnn_metrics['f1'] = metrics.f1_score(y_test, rnn_preds_test)
rnn_metrics['f1_macro'] = metrics.f1_score(y_test, rnn_preds_test,
                                           average='macro')
rnn_metrics['auc'] = metrics.roc_auc_score(y_test, rnn_preds_test)

for metric in rnn_metrics:
    print('{metric_name}: {metric_value}'.format(metric_name=metric, metric_value=rnn_metrics[metric]))

# Get training and test loss histories
rnn_training_loss = rnn_history.history['loss']
rnn_test_loss = rnn_history.history['val_loss']

# Get training and test accuracy history.
rnn_training_acc = rnn_history.history['accuracy']
rnn_test_acc = rnn_history.history['val_accuracy']

# Create count of the number of epochs
epoch_count = range(1, len(rnn_training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, rnn_training_loss, 'r--')
plt.plot(epoch_count, rnn_test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Visualize acc history
plt.plot(epoch_count, rnn_training_acc, 'r--')
plt.plot(epoch_count, rnn_test_acc, 'b-')
plt.legend(['Training Acc', 'Test Acc'])
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.show()
