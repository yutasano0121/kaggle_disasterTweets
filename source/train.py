# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.import tensorflow as tf

import tensorflow as tf
import argparse
import os
import numpy as np
import pandas as pd
import json


def macro_soft_f1(y_true, y_pred):
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.
    
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    tp = tf.reduce_sum(y_pred * y_true, axis=0)
    fp = tf.reduce_sum(y_pred * (1 - y_true), axis=0)
    fn = tf.reduce_sum((1 - y_pred) * y_true, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels
    
    return macro_cost


def model(x_train, y_train, x_test, y_test):
    """Generate a simple model"""
    # Set an early stopping.
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    
    # Set a weight initializer.
    xavier = tf.keras.initializers.GlorotNormal()
    
    # Define a model.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1024, input_dim=args.input_dim, activation=tf.nn.relu, kernel_initializer=xavier),
        tf.keras.layers.Dropout(args.drop_rate),
        tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_initializer=xavier),
        tf.keras.layers.Dropout(args.drop_rate),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    # Set an optimizer.
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    
    # Compile the model.
    model.compile(
        optimizer=opt,
        loss=macro_soft_f1,
        metrics=['accuracy']
    )
    
    model.fit(
        x_train, 
        y_train,
        epochs=args.epochs, 
        batch_size=args.batch_size,
        validation_data=(x_test, y_test),
        callbacks=[callback]
    )
    
    results = model.evaluate(x_test, y_test)
    print('test loss, test acc:', results)
    
    return model


def _load_training_data(base_dir):
    train = pd.read_csv(os.path.join(base_dir, 'train_processed_split.csv'), header=None)
    y_train = train[0].values
    x_train = train.drop(0, axis=1).to_numpy()
    return x_train, y_train


def _load_validation_data(base_dir):
    val = pd.read_csv(os.path.join(base_dir, 'validation.csv'), header=None)
    y_val = val[0].values
    x_val = val.drop(0, axis=1).to_numpy()
    return x_val, y_val


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm_model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current_host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    
    parser.add_argument('--input_dim', type=int, default=5223)
    parser.add_argument('--drop_rate', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()
    
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    train_data, train_labels = _load_training_data(args.train)
    eval_data, eval_labels = _load_validation_data(args.train)

    classifier = model(train_data, train_labels, eval_data, eval_labels)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
        classifier.save(os.path.join(args.sm_model_dir, '000000001'), 'my_model.h5')
