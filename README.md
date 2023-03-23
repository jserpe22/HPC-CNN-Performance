# Convolutional Neural Network Performance CPU versus GPU

## Abstract
Machine learning (ML) applications require the ability to process large datasets, and taking
advantage of using many threads to process this data synchronously is extremely important.
GPU’s have come to dominate in deep learning applications due to their massive multithreading
capabilities, which allow them to distribute processing. As datasets used in ML become
increasingly large in size and complexity, training models becomes more costly, both in time and
energy usage. Thus, it will be important to take advantage of hardware that minimizes these
costs.
Using TensorFlow a Convolutional Neural Network (CNN) model was developed to evaluate the
training time, testing time, and model performance of multiple CPUs and GPUs, across four
image classification datasets. Overall, the GPU’s were found to be much faster in both training
and testing, while appearing to generate more accurate models. This is in agreement with current
industry usage and topics discussed in class.

## Outline
This paper is divided into the following sections:
1. Hardware Description – a basic description of the CPU’s and GPU’s tested. This will
include a comparison of their marketed computational capabilities.
2. Software Description – an overview of the software used for the development and
execution of the evaluation.
3. Dataset Description – an overview of the four datasets used. These vary in the number of
training/test samples, initial image resolution, and number of classes.
4. Model Description – a brief explanation of the model used for testing and some relevant
CNN background information.
5. Results/Analysis – summary of the results for each platform and dataset.
6. Conclusions/Future Work – review of findings and discussion of next steps, including
overview of distributed learning techniques. 

## 1. Hardware Description
Four different processing units, two CPU’s and two GPU’s were evaluated for each dataset.
These were chosen due to their varying capabilities. All of the software was run on discovery on
nodes that had the following hardware specifications.
CPU Comparison [1]:

Both nodes had two sockets, so twice the number of cores shown were available for testing. The
AMD processor has much better potential, which is expected on a 5 year younger processor.
GPU Comparison [2],[3],[4],[5]: 

Besides being 4 years newer, the A100 also features nearly double the number of CUDA cores,
and over 2x the DP performance. A notable improvement between the P100 and later Nvidia
cards (including the A100) is the addition of tensor cores. Tensor cores are specialized
processing units that improve the efficiency of matrix multiplication on the GPU. This is
especially valuable to ML applications, which rely heavily on matrix multiplication as a base
calculation. Additionally, while regular CUDA cores can perform one operation per cycle, tensor
cores can perform multiple [6].

## 2. Software Description
The main library used for the development and testing of the model was TensorFlow [7].
TensorFlow provides many built in methods to interact with datasets, customize and compile 
various types of models, and functions to train and test models, as well as record valuable
metrics.
TensorFlow also provides easy API’s to download datasets and extract the training and testing
data. This makes it much easier to manage the datasets used for testing (rather than manually
downloading them and importing them into python from directories).
Specific functions used:
• tfds.load(dataset, with_info=True, as_supervised=True): Allows you to pass in the
dataset name as a string. Can be used to load any provided TensorFlow dataset.
• tf.image.resize(image, (IMAGE_RES, IMAGE_RES)): Allows you to reformat images to
have a consistent size. This is very important so that data input into the models are
consistent.
• tf.keras.Sequential([layers]): This is the main function used to develop a CNN. It can be
passed a list of calls to tf.keras.layers, where the user can specify parameters to easily
create a CNN model.
• compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True).
optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy']): Function to compile the
given model. Can specify the loss, optimizer, and metric used to optimize the model,
among other things.
• fit: function called to train the model. Accepts the train dataset and number of epochs to
train, among other things.
• evaluate: function to measure the performance of the model (loss, accuracy) given a test
dataset.
TensorFlow also has built in capabilities for distributed training techniques, which will be
discussed in the future work section.


## 3. Dataset Description
Four different datasets were tested to compare the impact of varying training/test sizes, image
resolution, and complexity of the data (number of classes). Each dataset belonged to the
TensorFlow Datasets library, which made it easy to import the data to discovery and use it with
the TensorFlow API, but other datasets could have been used with minimal changes.
The datasets are summarized in the following chart, ordered by the train size:

beans: This is a collection of images of bean leaves that belong to three classes. One is healthy,
and two show evidence of diseases, specifically Angular Leaf Spot and Bean Rust. The national
Crops Resources Research Institute (NaCRRI) in Uganda annotated the images, and the data was
collected by Makerere AI [https://www.tensorflow.org/datasets/catalog/beans]. 


uc_merced: This is a collection of aerial images of various urban areas, depicting different
classes (such as beaches, buildings, and highways). The images were extracted from the USGS
National Map Urban Area Imagery collection
[https://www.tensorflow.org/datasets/catalog/uc_merced]


TF Flowers: This is a simple dataset of 5 different species of flowers (5 classes)
[https://www.tensorflow.org/datasets/catalog/tf_flowers]. 


oxford_flowers: This is another flowers dataset, but was chosen in addition because it has a
much larger data set size, and a much larger variety of species (102 classes)
[https://www.tensorflow.org/datasets/catalog/oxford_flowers102].


In the results section, we will see how the data set size and complexity impacts the timing and
performance of the model. 


## 4. Model Description
As a general disclaimer, the model generated for this paper was not meant to be highly accurate
or effective in terms of the classification of the datasets. Its purpose was simply to explore the
performance of the different hardware chosen. In general models need to be tuned and
specialized for each dataset, which would have created too much variability in what was being
tested. To keep the results more consistent and clear, a general model was created and used for
each dataset and system. The model outlined in [12] was used as a starting point.
The type of model chosen was a Convolutional Neural Network (CNN). CNN’s are typically
used for image processing or classification, and contains one or more convolutional layers [14].
A convolution, mathematically, is the following: 


This can be broken down into matrix multiplication. Within a neural network, a convolutional
layer acts as a filter. The input data passes through this filter and produces output data. The basic
idea is that a given pixels position, and neighboring pixels, contain some information that is
relevant.

The other layers involved in the network are the max pooling layers, which take the maximum
value from a previous layer, and fully connected layers, which flatten previous data before it is
classified.

All layers of the model used are showed in the following image:


## 5. Results/Analysis
Timing data was recorded for both training and evaluating the model for each dataset on each
processor, along with training accuracy and loss, as well as test accuracy and loss. For each
run, the model was trained using the given training set over 7 epochs (the data was passed
through the model 7 times in total).
A summary of the results are shown in tabular form below, and displayed graphically in the
following subsections. 


Execution Time:
Timing data was taken for both training and testing the model. 


In the above graph, it is immediately evident how useful GPU’s are for these types of
workloads. Both the P100 and A100 perform several orders of magnitude better than the two
CPUs. The GPU’s take merely seconds to train the model, while the CPU’s can take up to 30
minutes. It is also clear that the larger the dataset, the longer training will take. 


For testing the data, there is a similar benefit to using the GPU’s, although the difference here is
much less. All the processors were able to validate the datasets within less than 30 seconds.
However, the GPU’s do not take more than 5 seconds for any dataset, while the testing time on
the CPU jumps considerably as the data set size increases. 


The reasoning for both of these trends is due to the nature of the neural network itself. As
discussed, the basic arithmetic involved in a CNN is matrix multiplication. As we have discussed
thoroughly, matrix multiplication can be optimized on the GPU due to the thousands of cores
available, and the ability to have many threads working concurrently. The limited threading
capabilities of the CPU cannot match that of the GPU, which leads to much slower training. 


Accuracy:
The accuracy of the model was recorded using both the training and testing datasets. 



For both the training and testing accuracy, there are a couple of important things to consider.
Firstly, it appears evident in both graphs that the platform does not have an effect on either
the training or testing accuracy. While there is some variation in the recorded data, this is
well within what would be expected as a result of shuffling the data when training the model.
Additionally, there should not be any difference in the accuracy, as the model is not changing
between the platforms, and thus should be outputting similar results.


CPU Comparison:
Between the CPU’s there is a noticeable difference in performance, shown here:


Overall, the 64 total cores of the two EPYC-7543s have better performance than the 28 total
cores of the two E5-2680s. Especially for smaller datasets, the speedup provided by the
AMD CPU is quite impressive. As the dataset size grows, the gap in performance narrows. In
both cases, the training time grew at a linear rate in terms of the number of training samples,
and these are all relatively small datasets that could train a model within 30 minutes. As n
becomes millions of rows of data, training on a CPU could take days or weeks, which is
impractical (when GPUs can perform much better). 

GPU Comparison:


Again, one GPU is clearly more powerful than the other. The A100 has around twice the
number of cores available, and roughly twice the overall processing power (including 432
tensor cores). These extra resources lead to a nice speedup, that within this range of data,
actually increases as the amount of data increases. This would start to decline as the data size
became large enough. Just like for the CPU’s, the training time grows roughly linearly as the
number of samples grows. So while training took less than a minute for each dataset on both
of these devices, much larger datasets, which are used in many applications, could still lead
to long training times on a single GPU. 


## 6. Conclusions/Future Work
In testing the training of a CNN over various datasets, it was clear that GPU’s are far superior to
CPU’s for this purpose. Training time was orders of magnitude lower on the GPU’s, thanks to
the massive amount of concurrency possible on a device with thousands of cores. Even
comparing just the two CPU’s, having more available cores can significantly reduce the time to
train. While CPU’s were still capable of training using these particular, relatively small datasets,
within a reasonable amount of time, the linear rate of growth in training time suggests that this
would quickly become unreasonable.
As datasets continue to grow, it will be important for developers to take advantage of all
processing power possible to reduce the time to train and test models. Given the results discussed
previously, even the capabilities of a single GPU, though much larger than a CPU, could start to
be limiting with a large enough dataset. As we have discussed, working on problems across
multiple processors and multiple nodes is not only possible, but can provide additional speedup,
given the computational cost outweighs the cost to communicate between nodes. 


Initially, my goal for this project was to test this very tradeoff by analyzing the performance of
training across multiple nodes, on both CPUs and GPUs. In my experimentation, I learned of a
couple methods to do this that are worth discussing here.
Distributed training using GPU’s is already the standard for large datasets, and is natively
supported in TensorFlow, and is quite easy to set up on the code side. TensorFlow allows you to
specify the strategy for training. One of which is a MirroredStrategy, which allows you to split
training across multiple GPUs or CPUs on the same machine. Another is
MultiWorkerMirroredStrategy, which allows you to split training across devices on multiple
machines. This involves setting up a cluster, which is a definition of the machines involved in the
computation, and the roles of each (worker, chief, etc). TensorFlow can automatically create a
cluster using the slurm configuration specified [15].
In addition to TensorFlows built in capabilities, third party libraries like Horovod can be used to
set up the distribution and split the dataset between different machines. Unlike TensorFlow,
which requires creating a cluster, Horovod handles the cluster creation for you and can be used
in conjunction with TensorFlow or PyTorch [16].
Testing and comparing these methods of distributed training, as well as different cluster
configurations, would provide interesting insight into how high the communication cost can
become, and how to optimize these workloads. This will become very important as workloads
become more demanding.


## Sources
[1] “AMD EPYC 7543 vs Intel Xeon E5-2680 V4.” GadgetVersus,
https://gadgetversus.com/processor/amd-epyc-7543-vs-intel-xeon-e5-2680-v4/.

[2] “Nvidia Tesla P100 Pcie 16 GB Specs.” TechPowerUp, 4 Nov. 2022,
https://www.techpowerup.com/gpu-specs/tesla-p100-pcie-16-gb.c2888.

[3] “Nvidia Tesla P100: The Most Advanced Data Center Accelerator.” NVIDIA,
https://www.nvidia.com/en-us/data-center/tesla-p100/.

[4] Nvidia A100 40GB Pcie GPU Accelerator. https://www.nvidia.com/content/dam/enzz/Solutions/Data-Center/a100/pdf/A100-PCIE-Prduct-Brief.pdf.

[5] “Nvidia A100 PCIe 40 GB Specs.” TechPowerUp, 4 Nov. 2022,
https://www.techpowerup.com/gpu-specs/a100-pcie-40-gb.c3623.

[6] Tensor Cores Explained: Do You Need Them? - Tech Centurion.
https://www.techcenturion.com/tensor-cores/.

[7] “Tensorflow.” TensorFlow, https://www.tensorflow.org/.

[8] Makerere Ai Lab. Beans disease dataset. January 2022, https://github.com/AI-LabMakerere/ibean/

[9] Yang, Yi and Newsam, Shawn. Bag-Of-Visual-Words and Spatial Extensions for Land-Use
Classification. 2010.

[10] The TensorFlow Team. Flowers. January 2019.
http://download.tensorflow.org/example_images/flower_photos.tgz

[11] Nilsback, M-E. and Zisserman, A. Automated Flower Classification over a Large Number of
Classes. December 2008.

[12] Gupta, Arijit. “Distributed Learning on Image Classification of Beans in Tensorflow.”
Medium, Towards Data Science, 7 May 2020,
https://towardsdatascience.com/distributed-learning-on-image-classification-of-beansin-tensorflow-5a85e6c3eb71.
[13] “Convolution.” Wikipedia, Wikimedia Foundation, 2 Dec. 2022,
https://en.wikipedia.org/wiki/Convolution.

[14] “Convolutional Neural Network.” Wikipedia, Wikimedia Foundation, 25 Nov. 2022,
https://en.wikipedia.org/wiki/Convolutional_neural_network.

[15] “Tf.distribute.experimental.multiworkermirroredstrategy : Tensorflow V2.11.0.”
TensorFlow,
https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMi
rroredStrategy.

[16] “Horovod with Tensorflow.” Horovod with TensorFlow - Horovod Documentation,
https://horovod.readthedocs.io/en/stable/tensorflow.html. 






