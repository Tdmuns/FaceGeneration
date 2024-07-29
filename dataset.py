import tensorflow as tf


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [128, 128])


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img


def setup_dataset(data_dir):
    list_ds = tf.data.Dataset.list_files(str(data_dir + '/*/*.jpg'), shuffle=True)
    return list_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE).batch(32)


def get_datasets():
    train_dataset = setup_dataset('data/train')
    validation_dataset = setup_dataset('data/validation')
    test_dataset = setup_dataset('data/test')
    return train_dataset, validation_dataset, test_dataset