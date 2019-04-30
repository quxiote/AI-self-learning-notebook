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
    # 获取本地图片，转换成numpy格式
    train_data, train_labels = load_image_dataset(csv_file_path=TRAIN_CSV_DIR, images_path=TRAIN_IMG_DIR)
    test_data, test_labels = load_image_dataset(csv_file_path=TEST_CSV_DIR, images_path=TEST_IMG_DIR)
 
    # 数据进行格式转换
    train_data = train_data.astype('float32') / 255
    test_data = test_data.astype('float32') / 255
    print("train data shape:", train_data.shape)
 
    # 使用图片识别器
    clf = ImageClassifier(verbose=True)
    # 给其训练数据和标签，训练的最长时间可以设定，假设为1分钟，autokers会不断找寻最优的网络模型
    clf.fit(train_data, train_labels, time_limit=1 * 60)
    # 找到最优模型后，再最后进行一次训练和验证
    clf.final_fit(train_data, train_labels, test_data, test_labels, retrain=True)
    # 给出评估结果
    y = clf.evaluate(test_data, test_labels)
    print("evaluate:", y)
 
    # 给一个图片试试预测是否准确
    img = load_img(PREDICT_IMG_PATH)
    x = img_to_array(img)
    x = x.astype('float32') / 255
    x = np.reshape(x, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
    print("x shape:", x.shape)
 
    # 最后的结果是一个numpy数组，里面是预测值4，意味着是马，说明预测准确
    y = clf.predict(x)
    print("predict:", y)
 
    # 导出我们生成的模型
    clf.export_keras_model(MODEL_DIR)
    # 加载模型
    model = load_model(MODEL_DIR)
    # 将模型导出成可视化图片
    plot_model(model, to_file=MODEL_PNG)