import numpy as np
import random
import torch as T


def pretty_print(*values):

    col_width = 13

    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)

    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))


def bucket_and_batch(data_X, data_Y, batch_size=2048):

    # print(data_X)

    data_X = data_X.tolist()
    data_Y = data_Y.tolist()

    idx = [id for id in range(len(data_X))]
    random.shuffle(idx)

    data_X = [data_X[id] for id in idx]
    data_Y = [data_Y[id] for id in idx]

    Y2Xnot = {}

    i = 0
    for i in range(len(data_X)):
        Y = data_Y[i]
        X = data_X[i]
        if Y not in Y2Xnot:
            Y2Xnot[Y] = []
            for X_, Y_ in zip(data_X, data_Y):
                if Y_ != Y:
                    Y2Xnot[Y].append((X_, Y_))

    batches_X = []
    batches_Y = []
    batches_X_counter = []
    batches_Y_counter = []

    i = 0
    while i < len(data_X):
        # print(i)
        batch_size_ = batch_size if (i+batch_size) <= len(data_X) else len(data_X)-i
        batch_X = []
        batch_Y = []
        batch_X_counter = []
        batch_Y_counter = []
        for j in range(i, i+batch_size_):
            batch_X.append(data_X[j])
            batch_Y.append(data_Y[j])

            data_list = Y2Xnot[data_Y[j]]
            # print(data_list[0:100])
            counter_X, counter_Y = random.choice(data_list)

            batch_X_counter.append(counter_X)
            batch_Y_counter.append(counter_Y)

        batch_X = T.tensor(batch_X).float().cuda()
        batch_Y = T.tensor(batch_Y).long().cuda()
        batch_X_counter = T.tensor(batch_X_counter).float().cuda()
        batch_Y_counter = T.tensor(batch_Y_counter).float().cuda()

        batches_X.append(batch_X)
        batches_Y.append(batch_Y)
        batches_X_counter.append(batch_X_counter)
        batches_Y_counter.append(batch_Y_counter)
        i += batch_size_

    return batches_X, batches_Y, batches_X_counter, batches_Y_counter
