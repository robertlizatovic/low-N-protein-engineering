import tensorflow as tf
import numpy as np
import pandas as pd
from unirep import babbler1900
import random
from typing import List
import sys
import argparse


def format_sequences(seq_file:str, formatted_file:str, model:babbler1900, use_stop:bool=False, max_len:int=275) -> None:
    with open(seq_file, "r") as source:
        with open(formatted_file, "w") as destination:
            for i,seq in enumerate(source):
                seq = seq.strip()
                if model.is_valid_seq(seq) and len(seq) < max_len: 
                    formatted = ",".join(map(str, model.format_seq(seq, stop=use_stop)))
                    destination.write(formatted)
                    destination.write('\n')

def format_dataset(seqs:List[str], model:babbler1900, max_len:int=275, use_stop:bool=True) -> List[List[int]]:
    """
    formats input sequences into integer lists using model's vocabulary
    
    returns a list of integer lists
    """
    seqs_fmt = []
    for s in seqs:
        s = s.strip()
        if model.is_valid_seq(s) and len(s) < max_len: 
            seqs_fmt.append(model.format_seq(s, stop=use_stop))
    return seqs_fmt

def batch_generator(data:List[List[int]], batch_size:int, shuffle:bool=True) -> np.array:
    """
    creates a batch generator over the input dataset that pads sequences
    to the length of the longest seq in a batch. Optionally shuffles the
    input dataset in place at the start. Remainders are dropped.
    -------------------------------
    data - list of integer lists
    
    returns np.array of shape (batch_size, max_len)
    """
    if shuffle:
        random.shuffle(data)
    data_size = len(data)
    assert data_size >= batch_size, "dataset must be larger than batch_size"
    n_batches = data_size // batch_size # will drop any remainders
    for i in range(n_batches):
        # make a batch
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch = data[start_idx:end_idx]
        max_len = max([len(s) for s in batch])
        batch_pad = np.zeros((batch_size, max_len), dtype=np.int32)
        for i, seq in enumerate(batch):
            batch_pad[i, 0:len(seq)] = seq
        yield batch_pad  

def calc_val_loss(model:babbler1900, sess:tf.Session, val_data:List[List[int]]) -> float:
    """computes the validation set loss. tf Variables should all be initialized within the session"""
    batch_sz = model._batch_size
    # get training operations
    _, loss, x_placeholder, y_placeholder, batch_size_placeholder, initial_state_placeholder = model.get_babbler_ops()
    val_losses = []
    for batch in batch_generator(val_data, batch_sz, shuffle=True):
        batch_x, batch_y = model.split_to_tuple(batch)
        loss_ = sess.run(loss,
                feed_dict={
                     x_placeholder: batch_x,
                     y_placeholder: batch_y,
                     batch_size_placeholder: batch_sz,
                     initial_state_placeholder: model._zero_state
                }
        )
        val_losses.append(loss_)
    return np.mean(val_losses)

def train_model(model:babbler1900, train_data:List[List[int]], val_data:List[List[int]], n_epochs:int=1, lr:float=1e-5, 
    shuffle:bool=True, ckpt_dir:str=None, val_freq:int=50, early_stopping:bool=True, patience:int=5):
    """trains the model using specified settings"""
    batch_sz = model._batch_size
    # get training operations
    _, loss, x_placeholder, y_placeholder, batch_size_placeholder, initial_state_placeholder = model.get_babbler_ops()
    # create an optimizer to fine-tune the model
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    fine_tuning_op = optimizer.minimize(loss)
    # train model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        i = 0 # global iteration
        es_counter = 0 # early stopping counter 
        min_val_loss = np.inf
        for e in range(n_epochs):
            print("Running epoch: %i" % (e + 1))
            # train on the training set batches
            for batch in batch_generator(train_data, batch_sz, shuffle=shuffle):
                i += 1
                batch_x, batch_y = model.split_to_tuple(batch)
                loss_, __, = sess.run([loss, fine_tuning_op],
                        feed_dict={
                             x_placeholder: batch_x,
                             y_placeholder: batch_y,
                             batch_size_placeholder: batch_sz,
                             initial_state_placeholder: model._zero_state
                        }
                )
                print("Iteration {0}: {1}".format(i, loss_))
                if i % val_freq == 0:
                    # calculate validation set performance
                    val_loss = calc_val_loss(model, sess, val_data)
                    print("Validation set loss: {}".format(val_loss))
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        es_counter = 0 # reset early stopping counter
                        if ckpt_dir is not None:
                            # store model weights
                            print("Checkpointing model weights")
                            model.dump_weights(sess, ckpt_dir)
                    elif early_stopping:
                        es_counter += 1
                        if es_counter >= patience:
                            print("No improvement in val loss for %i evaluations. Stopping training..." % patience)
                            return None
                    else:
                        continue
        print("Best validation set loss: %f" % min_val_loss)
    return None

def main(args):
    """trains the model using passed params"""
    # load train/val data
    train_data = pd.read_csv(args.train_data, header=None)
    val_data = pd.read_csv(args.val_data, header=None)

    # construct the model
    model = babbler1900(model_path=args.start_weights, batch_size=args.batch_sz)

    # format data
    train_fmt = format_dataset(train_data[0], model, use_stop=True)
    val_fmt = format_dataset(val_data[0], model, use_stop=True)
    print("Train set size: %i" % len(train_fmt))
    print("Val set size %i" % len(val_fmt))

    # train the model
    train_model(model, train_fmt, val_fmt, n_epochs=args.n_epochs, lr=args.lr, ckpt_dir=args.ckpt_dir, val_freq=args.val_freq, 
        early_stopping=args.early_stopping, patience=args.patience)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("start_weights", type=str, metavar="unirep_weights/1900_weights", help="starting weight dir")
    parser.add_argument("train_data", type=str, metavar="data/train_seq.txt", help="training seqs file")
    parser.add_argument("val_data", type=str, metavar="data/val_seq.txt", help="validation seqs file")
    parser.add_argument("n_epochs", type=int, metavar="N_EPOCHS", help="number of epochs to train")
    parser.add_argument("--batch_sz", type=int, default=256, metavar="256", help="batch size")
    parser.add_argument("--ckpt_dir", type=str, default=None, metavar="./eunirep_weights", help="dir to store fine-tuned weights")
    parser.add_argument("--lr", type=float, default=1e-5, metavar="0.00001", help="learning rate")
    parser.add_argument("--val_freq", type=int, default=50, metavar="50", help="validation/checkpoint frequency")
    parser.add_argument("--early_stopping", action="store_true", help="use early stopping")
    parser.add_argument("--patience", type=int, default=5, metavar="5", help="patience value for early stopping")
    args = parser.parse_args()
    main(args)
    sys.exit(0)
