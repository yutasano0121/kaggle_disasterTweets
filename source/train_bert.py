import numpy as np
import pandas as pd

import argparse
import os
import json
import pickle

#import matplotlib.pyplot as plt
#import seaborn as sns

import tokenization

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback


bert_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1', trainable=True)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

class ClassificationReport(Callback):
    
    def __init__(self, train_data=(), validation_data=()):
        super(Callback, self).__init__()
        
        self.X_train, self.y_train = train_data
        self.train_precision_scores = []
        self.train_recall_scores = []
        self.train_f1_scores = []
        
        self.X_val, self.y_val = validation_data
        self.val_precision_scores = []
        self.val_recall_scores = []
        self.val_f1_scores = [] 
               
    def on_epoch_end(self, epoch, logs={}):
        train_predictions = np.round(self.model.predict(self.X_train, verbose=0))        
        train_precision = precision_score(self.y_train, train_predictions, average='macro')
        train_recall = recall_score(self.y_train, train_predictions, average='macro')
        train_f1 = f1_score(self.y_train, train_predictions, average='macro')
        self.train_precision_scores.append(train_precision)        
        self.train_recall_scores.append(train_recall)
        self.train_f1_scores.append(train_f1)
        
        val_predictions = np.round(self.model.predict(self.X_val, verbose=0))
        val_precision = precision_score(self.y_val, val_predictions, average='macro')
        val_recall = recall_score(self.y_val, val_predictions, average='macro')
        val_f1 = f1_score(self.y_val, val_predictions, average='macro')
        self.val_precision_scores.append(val_precision)        
        self.val_recall_scores.append(val_recall)        
        self.val_f1_scores.append(val_f1)
        
        print('\nEpoch: {} - Training Precision: {:.6} - Training Recall: {:.6} - Training F1: {:.6}'.format(epoch + 1, train_precision, train_recall, train_f1))
        print('Epoch: {} - Validation Precision: {:.6} - Validation Recall: {:.6} - Validation F1: {:.6}'.format(epoch + 1, val_precision, val_recall, val_f1))  
        
        
def encode(texts, max_seq_length=128):

    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)
        text = text[:max_seq_length - 2]
        input_sequence = ['[CLS]'] + text + ['[SEP]']
        pad_len = max_seq_length - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_seq_length

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
        

def build_model(max_seq_length=128):

    input_word_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = Input(shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    segment_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name='segment_ids')    

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])   
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    optimizer = SGD(learning_rate=args.lr, momentum=0.8)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def train(X, max_seq_length=128):
    x_train_all = X['text_cleaned'].values
    y_train_all = X['target'].values

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_all, 
        y_train_all, 
        test_size=0.4, 
        random_state=0
    )

    x_train = encode(x_train, max_seq_length)
    x_val = encode(x_val, max_seq_length)


    # callback for metrics
    callback_metrics = ClassificationReport(train_data=(x_train, y_train), validation_data=(x_val, y_val))

    # callback to save intermediate weights
    checkpoint_path = os.path.join(args.sm_model_dir, "cp.ckpt")
    callback_cp = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, 
        save_weights_only=True,
        verbose=1
    )
    
    # callback for early stopping
    callback_es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)

    # Model
    model = build_model(max_seq_length)   

    model.fit(
        x_train, 
        y_train, 
        validation_data=(x_val, y_val), 
        #callbacks=[callback_metrics, callback_cp], 
        callbacks=[callback_metrics, callback_es],
        epochs=args.epochs, 
        batch_size=args.batch_size
    )

    score = {
        'train': {
            'precision': callback_metrics.train_precision_scores,
            'recall': callback_metrics.train_recall_scores,
            'f1': callback_metrics.train_f1_scores                    
        },
        'validation': {
            'precision': callback_metrics.val_precision_scores,
            'recall': callback_metrics.val_recall_scores,
            'f1': callback_metrics.val_f1_scores                    
        }
    }

    print(score)
    
    return model
"""           
def plot_learning_curve(self):

    fig, axes = plt.subplots(nrows=K, ncols=2, figsize=(20, K * 6), dpi=100)

    for i in range(K):

        # Classification Report curve
        sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.models[i].history.history['val_accuracy'], ax=axes[i][0], label='val_accuracy')
        sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.scores[i]['validation']['precision'], ax=axes[i][0], label='val_precision')
        sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.scores[i]['validation']['recall'], ax=axes[i][0], label='val_recall')
        sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.scores[i]['validation']['f1'], ax=axes[i][0], label='val_f1')        

        axes[i][0].legend() 
        axes[i][0].set_title('Fold {} Validation Classification Report'.format(i), fontsize=14)

        # Loss curve
        sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.models[0].history.history['loss'], ax=axes[i][1], label='train_loss')
        sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.models[0].history.history['val_loss'], ax=axes[i][1], label='val_loss')

        axes[i][1].legend() 
        axes[i][1].set_title('Fold {} Train / Validation Loss'.format(i), fontsize=14)

        for j in range(2):
            axes[i][j].set_xlabel('Epoch', size=12)
            axes[i][j].tick_params(axis='x', labelsize=12)
            axes[i][j].tick_params(axis='y', labelsize=12)

    plt.show()
"""


def _load_training_data(s3_path):
    fname = os.path.join(s3_path, 'train_for_bert.csv')
    train = pd.read_csv(fname)
    return train
    
    
def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm_model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current_host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()
    
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
      
    df_train = _load_training_data(args.train)

    model = train(df_train, args.max_seq_length)
    
    # save model to an S3 directory with version number '00000001'
    model.save(os.path.join(args.sm_model_dir, '00000001'), 'my_model.h5')

    #clf.plot_learning_curve()