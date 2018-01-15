---
title: Anatomy of TFRecord
date: 2018-01-14 19:27:49
tags:
---

# What is TFRecord?

TFRecord is a specific file format often used with TensorFlow, and it is their [recommended](https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data) format for dealing with huge data that does not fit in the memory. For this reason, Google uses this format for distributing some of their datasets, such as [the NSynth dataset](https://magenta.tensorflow.org/datasets/nsynth#files). The extension `.tfrecord` is used for these files.

# Why This Post?

I was frustrated that the documentation for using TFRecord files are especially lacking, when all I want to do is just read the numbers in the file. I'm writing this primarily for my own reference for dealing with TFRecord data, but hopefully will be helpful for others as well.

# TFRecord = Array of `Example`s

A TFRecord file contains an array of `Example`s.  `Example` is a data structure for representing a record, like an observation in a training or test dataset. A record is represented as a set of features, each of which has a name and can be an array of bytes, floats, or 64-bit integers. To summarize:

- An `Example` contains `Features`.
- `Features` is a mapping from the feature names stored as `string`s to `Feature`s.
- A `Feature` can be one of `BytesList`, `FloatList`, or `Int64List`.

These relations are defined in [example.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto) and [feature.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/feature.proto) in the TensorFlow source code, along with extensive comments. As the extension `.proto` suggests, these definitions are based on protocol buffers.

# Why Protocol Buffers?

Google's [Protocol buffers](https://developers.google.com/protocol-buffers/)﻿ are a serialization scheme for structured data. In other words, protocol buffers are used for serializing structured data into a byte array or parsing the serialized array as the original data, so that they can be sent over the network or stored as a file. In this sense, it is similar to JSON, XML, or [MessagePack](https://msgpack.org/).

Unlike JSON, protocol buffers can only work with the messages whose schema is predefined, using the [protocol buffer language](https://developers.google.com/protocol-buffers/docs/proto3). The protocol buffer compiler, or `protoc`, uses the language-agnostic `.proto` files to generate the optimized code for the serializer and the parser, in the supported languages including C++, Python, and Java.

With the cost of having to use the definition files and tooling, protocol buffers can offer a lot faster processing speed compared to text-based formats like JSON or XML. Most importantly, it doesn't make sense to store multimedia data like images or audio in a text-based format. These data need to be stored in large multidimensional numeric arrays, or tensors, and using protocol buffers the content of the file can be directly copied to the system memory and interpreted as tensors without any text processing for the numbers.

In addition to the efficient binary storage, an `Example` in TFRecord can contain other simple features like categories or labels, often represented using a single number or a string. So, thanks to the TFRecord format being based on protocol buffers, we can use a single streamlined format for storing both some high-dimensional data and the simple metadata. It's time for a goodbye to the old days dealing with the heap of media files and the metadata files separately!

# The Low-Level Way

The internals for reading the TFRecord format is implemented in C++, in the [RecordReader class](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/io/record_reader.cc). This part is not dealing with protocol buffers yet, but it is just slicing the file into many chunks, each of them being a serialized protocol buffer `Example`. Internally it uses a 12-byte header and 4-byte footer to store the length of each chunk with the checksums, and also supports optional GZIP compression.

On the Python side, `tf.python_io.tf_record_iterator` provides a way to iterate over the serialized `Example`s in the TFRecord files. Each time the iterator spits a bytes object, which then can be parsed using the protocol buffer class `Example`.

```python
import tensorflow as tf
import numpy as np

for record in tf.python_io.tf_record_iterator("nsynth-valid.tfrecord"):
    example = tf.train.Example()
    example.ParseFromString(record) # calling protocol buffer API
    
    audio = np.array(example.features.feature['audio'].float_list.value)
    pitch = example.features.feature['pitch'].int64_list.value[0]
    
    print(len(audio), pitch)
```

Being a generated protocol buffer message class, `tf.train.Example` supports `ParseFromString` method that parses given bytes and populates the corresponding fields. This way, although not very concise, we can directly access the numbers in the TFRecord file without having to deal with the usual TensorFlow boilerplates like `tf.Session`s or `tf.Tensor`s.

**A note for MacOS**: This code runs very slow on Macs, because the MacOS versions of the protocol buffers Python package does not ship with the native library by default. It seems that [Mac users can manually build the package](https://github.com/tensorflow/tensorflow/issues/15417) for the maximum performance.

# The Canonical Way

Until TensorFlow 1.2, the recommended way for dealing with the influx of data was to use the [multithreading and queues](https://www.tensorflow.org/api_guides/python/threading_and_queues). However, with the `tf.data` package becoming official in TensorFlow 1.4, now the recommended way is to use [the Dataset API](https://www.tensorflow.org/programmers_guide/datasets). So whenever you see the word `queue`, you may assume that the code is using a deprecated way of dealing with datasets.

The [fully-connected MNIST example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py) shows how to read a TFRecord file using the Dataset API, and the `inputs` function contains the core part:

```python
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.repeat(num_epochs)

    dataset = dataset.map(decode)
    dataset = dataset.map(augment)
    dataset = dataset.map(normalize)

    dataset = dataset.shuffle(1000 + 3 * batch_size)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
  return iterator.get_next()
```

What this excerpt does in each line are:

1. Create a `tf.data.TFRecordDataset` instance using the file name(s).
2. By calling `dataset.repeat(num_epochs)`, make everything repeat up to the number of epochs.
3. By calling `dataset.map(decode)`, parse the bytes into a tuple of the image and the label:
   * In the `decode` function, call `tf.parse_single_example()`, where:
     * The first argument is the raw bytes, and
     * The second argument is a dictionary mapping the feature names and types.
4. Make a few additional `map` calls for data augmentation and normalization.
5. Shuffle the dataset using `dataset.shuffle(buffer_size)`:
   * The parameter is the size of the buffer from which the random sample is selected.
6. Make batches using `dataset.batch(batch_size)`:
   * Each `tf.Tensor` in the dataset is prepended with an additional dimension for the batch.
7. Create an iterator for the resulting `dataset` using `make_one_shot_iterator()`.
8. Return the TensorFlow operation `get_next()`, for getting the next iteratee.

These are the standard steps for using the new Dataset API, which allows functional APIs to transform the dataset in a very modular fashion. For example, this make it very easy to toggle the data augmentation or shuffling, or plug a different set of augmentation methods.

# Taking The Best of Both Worlds

If we want to keep using the goodies that Keras like its cleaner API and toolings, we can convert the tensorflow operation into a Python generator that can be fed to `fit_generator()`. This way, rather than starting from the scratch as in the low-level way above, we can benefit from the functional Dataset API for easier data transformation and memory management for large datasets, while keeping using what Keras is good for.

Below is a full example for loading a TFRecord file and converting it to a generator producing the usual (data, label) tuples:

```python
def inputs(files, batch_size=32):
    dataset = tf.data.TFRecordDataset(files).repeat()

    dataset = dataset.map(lambda record: tf.parse_single_example(record, features={
        'audio': tf.FixedLenFeature([64000], tf.float32),
        'note': tf.FixedLenFeature([], tf.int64)
    }))
    dataset = dataset.map(lambda features: (features['audio'], features['note']))

    dataset = dataset.shuffle(8000).batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def to_generator(inputs):
    from keras import backend as K
    
    sess = K.get_session()
    while True:
        yield sess.run(inputs)
```

In summary:

* TFRecord and the new Dataset API make a good combination for the simpler data input pipeline.
* You need to know the name and the type of the features to parse.
* You can continue using your Keras code with this new API.