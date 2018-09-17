import os
import tensorflow as tf

def gen_tfrecord_file(file_path):
    cwd = file_path
    classes = {'ball1', 'ball2'}  # 类别设定
    # 定义writer用于写入数据，tf.python_io.TFRecordWriter 写入到TFRecords文件中
    writer = tf.python_io.TFRecordWriter("ball_train.tfrecords")  # 定义生成的文件名为“ball_train.tfrecords”
    for index, name in enumerate(classes):
        class_path = cwd + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name  # 每一个图片的地址
            img = Image.open(img_path)
            img = img.resize((224, 224))  # 将图片保存成224×224大小
            img_raw = img.tobytes()  # 将图片转化为原生bytes，#tf.train.Example 协议内存块包含了Features字段，通过feature将图片的二进制数据和label进行统一封装
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))  # example对象对label和image进行封装
            writer.write(example.SerializeToString())  # 序列化为字符串，example协议内存块转化为字符串
    writer.close()

if __name__ == "__main__":
    gen_tfrecord_file(file_path)