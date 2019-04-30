import os
import csv
 
TRAIN_IMG_DIR = '/home/yourname/Documents/tensorflow/images/500pics2/train'
TRAIN_CSV_DIR = '/home/yourname/Documents/tensorflow/images/500pics2/train_labels.csv'
TEST_IMG_DIR = '/home/yourname/Documents/tensorflow/images/500pics2/test'
TEST_CSV_DIR = '/home/yourname/Documents/tensorflow/images/500pics2/test_labels.csv'
 
def mkcsv(img_dir, csv_dir):
    list = []
    list.append(['File Name','Label'])
    for file_name in os.listdir(img_dir):
        if file_name[0] == '3':   #bus
            item = [file_name, 0]
        elif file_name[0] == '4': #dinosaur
            item = [file_name, 1]
        elif file_name[0] == '5': #elephant
            item = [file_name, 2]
        elif file_name[0] == '6': #flower
            item = [file_name, 3]
        else:
            item = [file_name, 4] #horse
        list.append(item)
 
    print(list)
    f = open(csv_dir, 'w', newline='')
    writer = csv.writer(f)
    writer.writerows(list)
 
mkcsv(TRAIN_IMG_DIR, TRAIN_CSV_DIR)
mkcsv(TEST_IMG_DIR, TEST_CSV_DIR)