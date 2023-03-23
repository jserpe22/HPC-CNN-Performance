import tensorflow_datasets as tfds
import tensorflow as tf
import PIL


# Code influenced by the example here: https://towardsdatascience.com/distributed-learning-on-image-classification-of-beans-in-tensorflow-5a85e6c3eb71
# Author: Arijit Grupta
# Date: May 7th, 2022
# Title: Distributed Learning on Image Classifcation of Beans in TensorFlow

IMAGE_RES = 224

def scale(image, label):
    #image=tf.cast(image, tf.float32)
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))
    image /= 255
    return image,label

def getDatasets(buffer_size, batch_size, dataset):
    if dataset == 'beans':
        #datasets, info = tfds.load(dataset, with_info=True, as_supervised=True)
        datasets, info= tfds.load(dataset, with_info=True, as_supervised=True) 
        train, test = datasets['train'], datasets['test']
    elif dataset == 'oxford_flowers102':
        datasets, info = tfds.load(dataset,with_info=True, as_supervised=True)
        test, train = datasets['train'], datasets['test']  
    else:    
        (train, test), info = tfds.load(dataset, split=['train[:70%]', 'train[70%:]'],with_info=True, as_supervised=True)
    #train, test = datasets['train'], datasets['test']
    #print(info)
    num_train_examples = len(train)
    num_test_examples = len(test)
    train_dataset = train.map(scale).cache().shuffle(buffer_size).batch(batch_size)
    eval_dataset = test.map(scale).batch(batch_size)
    print("Using dataset: ", dataset)
    print(train_dataset)
    print(eval_dataset)
    num_classes = info.features['label'].num_classes
    print("Number of Classes: ", num_classes)
    print("num_train: ", num_train_examples, " num_test: ", num_test_examples)
    return train_dataset, eval_dataset, num_classes

def buildModel(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, activation = 'relu', input_shape=(IMAGE_RES, IMAGE_RES, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation = 'relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation = 'relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(num_classes)
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy'])
    return model


def getModel(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, activation = 'relu', input_shape=(IMAGE_RES, IMAGE_RES, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation = 'relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation = 'relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(num_classes)
    ])
    return model
