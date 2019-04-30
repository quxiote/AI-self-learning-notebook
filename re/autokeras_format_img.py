from tensorflow.keras.preprocessing import image
import os

TEST_IMG_DIR_INPUT = "/home/yourname/Documents/tensorflow/images/500pics2/test_origin"
TEST_IMG_DIR_OUTPUT = "/home/yourname/Documents/tensorflow/images/500pics2/test"
TRAIN_IMG_DIR_INPUT = "/home/yourname/Documents/tensorflow/images/500pics2/train_origin"
TRAIN_IMG_DIR_OUTPUT = "/home/yourname/Documents/tensorflow/images/500pics2/train"
IMAGE_SIZE = 28

def format_img(input_dir, output_dir):
    for file_name in os.listdir(input_dir):
        path_name = os.path.join(input_dir, file_name)
        img = image.load_img(path_name, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        path_name = os.path.join(output_dir, file_name)
        img.save(path_name)

format_img(TEST_IMG_DIR_INPUT, TEST_IMG_DIR_OUTPUT)
format_img(TRAIN_IMG_DIR_INPUT, TRAIN_IMG_DIR_OUTPUT)