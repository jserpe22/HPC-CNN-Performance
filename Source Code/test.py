import model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import argparse
import sys
import time
import energyusage


parser = argparse.ArgumentParser(description='MultiGPU/CPU Training',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--dataset', action='store', default='beans', choices=['beans', 'uc_merced', 'tf_flowers', 'oxford_flowers102'], help='sets tfds dataset to use')

args = parser.parse_args()
args.cuda = not args.no_cuda
dataset = args.dataset
print("Using dataset: ", dataset)


if args.cuda:
    print("USING GPUs")
else:
    print("DISABLING GPUS")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
buffer_size = 10000

strategy = tf.distribute.MirroredStrategy()

batch_size_per_replica = 32
batch_size = batch_size_per_replica*strategy.num_replicas_in_sync

train, test, num_classes = model.getDatasets(buffer_size, batch_size, dataset)

#print("train: ", train)
#print("train[0]", train[0])
#print("train[1]", train[1])

with strategy.scope():
    model = model.buildModel(num_classes)

start = time.time()
#energyusage.evaluate(model.fit(train, epochs=7))
model.fit(train,epochs=7)
end = time.time()
print("Time to train model: ", end-start, " seconds")
start = time.time()
model.evaluate(test)
end = time.time()
print("Time to test model: ", end-start, "seconds")

#tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

#energyusage.evaluate(run) 
