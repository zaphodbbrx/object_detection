import io

import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from sklearn.utils import compute_class_weight

from object_detection.utils import dataset_util

from conf import *


flags = tf.app.flags
flags.DEFINE_string('output_path', 'data/train.record', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def is_small_object(pack):
    xmin, xmax, ymin, ymax, cl, cl_text = pack
    return (xmax - xmin) * (ymax - ymin) > 16*16


def create_tf_example(fname: str, height: int, width: int, df: pd.DataFrame, weights_dct: dict):
    # TODO(user): Populate the following variables from your example.

    with tf.gfile.GFile(fname, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_image_data = io.BytesIO(encoded_jpg)  # Encoded image bytes
    image_format = b'jpeg'  # b'jpeg' or b'png'

    xmins = (df.xmin / width).tolist()  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = (df.xmax / width).tolist()  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = (df.ymin / height).tolist()  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = (df.ymax / height).tolist()  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = df['class'].tolist()  # List of string class name of bounding box (1 per box)
    classes = [class2num[cl] for cl in classes_text]  # List of integer class id of bounding box (1 per box)
    #xmins, xmaxs, ymins, ymaxs, classes, classes_text = zip(*filter(is_small_object, zip(xmins, xmaxs, ymins, ymaxs, classes, classes_text)))
    weights = [weights_dct[cl] for cl in classes_text]
    classes_text = [cl.encode('utf8') for cl in classes_text]
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(fname.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(fname.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(list(xmins)),
        'image/object/bbox/xmax': dataset_util.float_list_feature(list(xmaxs)),
        'image/object/bbox/ymin': dataset_util.float_list_feature(list(ymins)),
        'image/object/bbox/ymax': dataset_util.float_list_feature(list(ymaxs)),
        'image/object/class/text': dataset_util.bytes_list_feature(list(classes_text)),
        'image/object/class/label': dataset_util.int64_list_feature(list(classes)),
        'image/object/weight': dataset_util.float_list_feature(weights)
    }))
    return tf_example


def main(csv_path: str):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    # TODO(user): Write code to read in your dataset to examples variable
    df = pd.read_csv(csv_path)
    weights = compute_class_weight('balanced', df['class'].unique().tolist(), df['class'].tolist())
    weights_dct = dict(zip(df['class'].unique().tolist(), weights))
    examples = df.groupby(['fname', 'imheight', 'imwidth'])
    for (fname, height, width), df in tqdm(examples):
        try:
            tf_example = create_tf_example(fname, height, width, df, weights_dct)
            writer.write(tf_example.SerializeToString())
        except ValueError:
            #image with only small object
            continue

    writer.close()


if __name__ == '__main__':
    #tf.app.run()
    main('data/train.csv')
