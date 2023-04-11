import cv2
import matplotlib.pyplot as plt
# image = cv2.imread("datasets/numbers/trainingSet/0/img_1.jpg")
# plt.imshow(image)
# plt.title("Image")

import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer
dir_path = "datasets/numbers/trainingSet/"
images = list()
labels = list()
for number in range(0, 10):
    folder = dir_path + str(number)
    for image_file in os.listdir(folder):
        image_gray = cv2.imread(os.path.join(folder, image_file), cv2.IMREAD_GRAYSCALE)
        image = np.array(image_gray).flatten()
        images.append(image/255)
        labels.append(number)
    print("[INFO] Processed {}/{}".format(number + 1, 10))
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

from sklearn.model_selection import train_test_split
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.1, random_state=42)

import random
num_clients = 16
client_names = ["client_{}".format(i + 1) for i in range(num_clients)]
data = list(zip(images, labels))
random.shuffle(data)
size = len(data)//num_clients
data_shards = [data[i:i + size] for i in range(0, size*num_clients, size)]
clients = {client_names[i]: data_shards[i] for i in range(num_clients)}

import math
num_helpers = 25

import tensorflow as tf
batch_size = 32
clients_batched = dict()
for (client_name, data_shard) in clients.items():
    data, labels = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(labels)))
    clients_batched[client_name] = dataset.shuffle(len(labels)).batch(batch_size)
test_batched = tf.data.Dataset.from_tensor_slices((images_test, labels_test)).batch(len(labels_test))

from keras.models import Sequential
from keras.layers import Dense, Activation
class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

from keras.optimizers import SGD
lr = 0.01 
comms_round = 100
loss='categorical_crossentropy'
metrics = ['accuracy']
optimizer = SGD(lr=lr, decay=lr/comms_round, momentum=0.9)

def weight_scaling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    return local_count/global_count

def scale_model_weights(weight, scalar):
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final

def sum_scaled_weights(scaled_weight_list):
    avg_grad = list()
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
    return avg_grad

def get_shape(weights):
    shapes = [i.shape for i in weights]
    return shapes

def flatten_model_weights(weights):
    flattened_weights = np.concatenate([i.flatten() for i in weights])
    return flattened_weights

def restore_model_shape(flattened_weights, shapes):
    weights = []
    index = 0
    for shape in shapes:
        size = np.product(shape)
        arr = np.array(flattened_weights[index : index + size])
        weights.append(arr.reshape(shape))
        index += size
    return weights

from bitstring import BitArray

def modulation(weights):
    weights = flatten_model_weights(weights)
    packets = []
    for i in range(len(weights)):
        packet = BitArray(float=weights[i], length=32)
        packets.append(packet.bin)
    return packets

def demodulation(weights, model_shape):
    return restore_model_shape(weights, model_shape)

import random

def calculateDecodingProbability(n, q, k):
    if n < k:
        return 0
    num, den = 1, 1
    for i in range(0, k):
        num *= (1 - q**(i - n))
        den *= (1 - q**(i - n + 1))
    return 1 - (1 - num)/(1 - den)

def transmissionCH(packet_loss_prob, q, k):
    succesfully_transmitted = 0
    transmissions = []
    transmissionCnt = 0
    for _ in range(num_helpers):
        transmissionCnt += 1
        if random.random() > packet_loss_prob:
            transmissions.append(1)
            succesfully_transmitted += 1
        else:
            transmissions.append(0)
    if succesfully_transmitted >= k:
        return transmissions, True
    while transmissionCnt < 2*num_helpers:
        transmissionCnt += 1
        if random.random() > packet_loss_prob:
            transmissions.append(1)
            succesfully_transmitted += 1
        else:
            transmissions.append(0)
        probability_success = calculateDecodingProbability(succesfully_transmitted, q, k)
        if random.random() < probability_success:
            return transmissions, True
    return transmissions, False

from keras.losses import CategoricalCrossentropy
from sklearn.metrics import accuracy_score
def test_model(X_test, Y_test,  model, comm_round):
    cce = CategoricalCrossentropy(from_logits=True)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round+1, acc, loss))
    return acc, loss

from keras import backend as K

def fedML(packet_loss_prob, q, k):
    smlp_global = SimpleMLP()
    global_model = smlp_global.build(784, 10)
    model_shape = get_shape(global_model.get_weights())
    accuracy = []
    for comm_round in range(comms_round):
        global_weights = global_model.get_weights()
        client_names= list(clients_batched.keys())
        scaled_weights_list = []
        comm_matrix = []
        random.shuffle(client_names)
        lost_clients = 0
        for _, client in enumerate(client_names):
            smlp_local = SimpleMLP()
            local_model = smlp_local.build(784, 10)
            local_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            local_model.set_weights(global_weights)
            local_model.fit(clients_batched[client], epochs=1, verbose=0)
            scaling_factor = weight_scaling_factor(clients_batched, client)
            scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
            bitWeights = modulation(scaled_weights)
            transmissions, decodable = transmissionCH(packet_loss_prob, q, k)
            comm_matrix.append(transmissions)
            if not decodable:
                lost_clients += 1
            else:
                weights = bitWeights
                for i in range(len(bitWeights)):
                    weights[i] = BitArray(bin=bitWeights[i]).float
                scaled_weights_list.append(weights)
            K.clear_session()
        if lost_clients != 0:
            compensated_lost_weights = scale_model_weights(global_weights, lost_clients*weight_scaling_factor(clients_batched, 'client_1'))
            packets = modulation(compensated_lost_weights)
            weights = packets
            for i in range(len(packets)):
                weights[i] = BitArray(bin=packets[i]).float
            scaled_weights_list.append(weights)
        global_weights = sum_scaled_weights(scaled_weights_list)
        global_weights = demodulation(global_weights, model_shape)
        global_model.set_weights(global_weights)
        with open('output/hybrid/P{}/comm_matrix.txt'.format(int(100*packet_loss_prob)), 'a') as f:
            f.write('Round ' + str(comm_round) + ': ' + str(comm_matrix) + '\n')
        for(X_test, Y_test) in test_batched:
            global_acc, _ = test_model(X_test, Y_test, global_model, comm_round)
            accuracy.append(global_acc)
    return accuracy

import sys
if __name__ == "__main__":
    p = float(sys.argv[1])
    q = 256
    k = 20
    acc = fedML(p, q, k)
    np.savetxt("./output/hybrid/P{}/acc.csv".format(int(100*p)), acc, delimiter=",")
