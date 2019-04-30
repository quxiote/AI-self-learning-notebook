from autokeras.image.image_supervised import load_image_dataset, ImageClassifier
from keras.models import load_model
from keras.utils import plot_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
 
TRAIN_CSV_DIR = '/home/yourname/Documents/tensorflow/images/500pics2/train_labels.csv'
TRAIN_IMG_DIR = '/home/yourname/Documents/tensorflow/images/500pics2/train'
TEST_CSV_DIR = '/home/yourname/Documents/tensorflow/images/500pics2/test_labels.csv'
TEST_IMG_DIR = '/home/yourname/Documents/tensorflow/images/500pics2/test'
 
PREDICT_IMG_PATH = '/home/yourname/Documents/tensorflow/images/500pics2/test/719.jpg'
 
MODEL_DIR = '/home/yourname/Documents/tensorflow/images/500pics2/model/my_model.h5'
MODEL_PNG = '/home/yourname/Documents/tensorflow/images/500pics2/model/model.png'
IMAGE_SIZE = 28
 
if __name__ == '__main__':
    # ��ȡ����ͼƬ��ת����numpy��ʽ
    train_data, train_labels = load_image_dataset(csv_file_path=TRAIN_CSV_DIR, images_path=TRAIN_IMG_DIR)
    test_data, test_labels = load_image_dataset(csv_file_path=TEST_CSV_DIR, images_path=TEST_IMG_DIR)
 
    # ���ݽ��и�ʽת��
    train_data = train_data.astype('float32') / 255
    test_data = test_data.astype('float32') / 255
    print("train data shape:", train_data.shape)
 
    # ʹ��ͼƬʶ����
    clf = ImageClassifier(verbose=True)
    # ����ѵ�����ݺͱ�ǩ��ѵ�����ʱ������趨������Ϊ1���ӣ�autokers�᲻����Ѱ���ŵ�����ģ��
    clf.fit(train_data, train_labels, time_limit=1 * 60)
    # �ҵ�����ģ�ͺ���������һ��ѵ������֤
    clf.final_fit(train_data, train_labels, test_data, test_labels, retrain=True)
    # �����������
    y = clf.evaluate(test_data, test_labels)
    print("evaluate:", y)
 
    # ��һ��ͼƬ����Ԥ���Ƿ�׼ȷ
    img = load_img(PREDICT_IMG_PATH)
    x = img_to_array(img)
    x = x.astype('float32') / 255
    x = np.reshape(x, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
    print("x shape:", x.shape)
 
    # ���Ľ����һ��numpy���飬������Ԥ��ֵ4����ζ������˵��Ԥ��׼ȷ
    y = clf.predict(x)
    print("predict:", y)
 
    # �����������ɵ�ģ��
    clf.export_keras_model(MODEL_DIR)
    # ����ģ��
    model = load_model(MODEL_DIR)
    # ��ģ�͵����ɿ��ӻ�ͼƬ
    plot_model(model, to_file=MODEL_PNG)