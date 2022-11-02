#dataset_provider


import tensorflow as tf
slim = tf.contrib.slim


def provide_data_queue(path, batch_size, subset, num_samples, num_readers=1,
                 num_threads=1):
    """Provides batches of MNIST digits.
    Args:
      split_name: Either 'train' or 'test'.
      batch_size: The number of images in each batch.
      dataset_dir: The directory where the MNIST data can be found.
      num_readers: Number of dataset readers.
      num_threads: Number of prefetching threads.
    Returns:
      images: A `Tensor` of size [batch_size, 28, 28, 1]
      one_hot_labels: A `Tensor` of size [batch_size, mnist.NUM_CLASSES], where
        each row has a single element set to one and the rprovide_data_queueest set to zeros.
      num_samples: The number of total samples in the dataset.
    Raises:
      ValueError: If `split_name` is not either 'train' or 'test'.
    """
    dataset_dir = path + subset + '.tfrecords'
    print(dataset_dir)
    reader = tf.TFRecordReader
    keys_to_features = {
        'X': tf.FixedLenFeature([128], tf.float32),
        'Y': tf.FixedLenFeature([1], tf.int64)
    }

    items_to_handlers = {
        'X': slim.tfexample_decoder.Tensor('X'),
        'Y': slim.tfexample_decoder.Tensor('Y')
    }

    items_to_descriptions = {
        'X': 'a 178X11 float32 array',
        'Y': 'a 178X2 int64 array'
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                      items_to_handlers)
    dataset = slim.dataset.Dataset(data_sources=dataset_dir, reader=reader,
                                   decoder=decoder, num_samples=num_samples,
                                   items_to_descriptions=items_to_descriptions)
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=num_readers,
        common_queue_capacity=2 * batch_size,
        common_queue_min=batch_size)
    [feature, label] = provider.get(['X', 'Y'])
    # Preprocess the images.
    # image = (tf.to_float(image) - 128.0) / 128.0
    #feature = feature[label[1] == 2]
    # Creates a QueueRunner for the pre-fetching operation.
    features, labels = tf.train.batch(
        [feature, label],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=5 * batch_size)

    one_hot_labels = tf.squeeze(tf.one_hot(labels,2))
    return features, one_hot_labels, dataset.num_samples

def provide_tf_queue(path, batch_size, subset, num_readers=1,
                 num_threads=1):
    """Provides batches of MNIST digits.
    Args:
      split_name: Either 'train' or 'test'.
      batch_size: The number of images in each batch.
      dataset_dir: The directory where the MNIST data can be found.
      num_readers: Number of dataset readers.
      num_threads: Number of prefetching threads.
    Returns:
      images: A `Tensor` of size [batch_size, 28, 28, 1]
      one_hot_labels: A `Tensor` of size [batch_size, mnist.NUM_CLASSES], where
        each row has a single element set to one and the rest set to zeros.
      num_samples: The number of total samples in the dataset.
    Raises:
      ValueError: If `split_name` is not either 'train' or 'test'.
    """
    real_shape = [64, 64, 1]
    dataset_dir = path + subset + '_set.tfrecords'
    reader = tf.TFRecordReader
    keys_to_features = {
        'X': tf.FixedLenFeature([real_shape[0]*real_shape[1]], tf.float32),
        'Y': tf.FixedLenFeature([2], tf.int64)
    }
    items_to_handlers = {
        'X': slim.tfexample_decoder.Tensor('X'),
        'Y': slim.tfexample_decoder.Tensor('Y')
    }

    items_to_descriptions = {
        'X': 'a 178X11 float32 array',
        'Y': 'a 178X2 int64 array'
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                      items_to_handlers)
    dataset = slim.dataset.Dataset(data_sources=dataset_dir, reader=reader,
                                   decoder=decoder, num_samples=4968,
                                   items_to_descriptions=items_to_descriptions)

    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=num_readers,
        common_queue_capacity=2 * batch_size,
        common_queue_min=batch_size)
    [feature, label] = provider.get(['X', 'Y'])

    # Preprocess the images.
    feature = tf.reshape(feature, real_shape)
    # image = (tf.to_float(image) - 128.0) / 128.0
    #feature = feature[label[1] == 2]
    # Creates a QueueRunner for the pre-fetching operation.
    features, labels = tf.train.batch(
        [feature, label[1]],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=5 * batch_size)

    one_hot_labels = tf.squeeze(tf.one_hot(labels, 2))
    return features, one_hot_labels, dataset.num_samples

def _parse_(serialized_example):
    feature = {'X': tf.FixedLenFeature([128], tf.float32),
               'Y': tf.FixedLenFeature([1], tf.int64)}
    example = tf.parse_single_example(serialized_example, features=feature)
    instance = example['X']
    label = tf.cast(example['Y'], tf.int8)#int8
    return instance, label #tf.one_hot(label[1],17)


def tfrecord_train_input_fn(tfrecord_path, batch_size=32):

    tfrecord_dataset = tf.data.TFRecordDataset(tfrecord_path)
    tfrecord_dataset = tfrecord_dataset.map(_parse_)
    tfrecord_dataset = tfrecord_dataset.batch(batch_size)
    tfrecord_iterator = tfrecord_dataset.make_one_shot_iterator()
    data, target = tfrecord_iterator.get_next()
    return data, target


def provide_data(path, batch_size=32, subset='train'):
    tfrecord_path = path + subset + '.tfrecords'
    print(tfrecord_path)
    return tfrecord_train_input_fn(tfrecord_path, batch_size)


def _parse_tf_(serialized_example):
    feature = {'X': tf.FixedLenFeature([64*64], tf.float32),
               'Y': tf.FixedLenFeature([2], tf.int64)}
    example = tf.parse_single_example(serialized_example, features=feature)
    instance = example['X']
    instance = tf.reshape(instance, [64,64])
    #instance = tf.reduce_mean(instance,1)
    label = tf.cast(example['Y'], tf.uint8)
    return instance, label[1]

def provide_tf(path, batch_size=32, subset='train'):
    tfrecord_path = path + subset + '_set.tfrecords'
    print(tfrecord_path)
    tfrecord_dataset = tf.data.TFRecordDataset(tfrecord_path)
    tfrecord_dataset = tfrecord_dataset.map(_parse_tf_)
    tfrecord_dataset = tfrecord_dataset.batch(batch_size)
    tfrecord_iterator = tfrecord_dataset.make_one_shot_iterator()
    data, target = tfrecord_iterator.get_next()
    return data, target