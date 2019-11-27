import os
from os import listdir
from os.path import isfile, join
import cv2
def labeling(data_dir):
    labels = os.listdir(data_dir)
    num_classes = len(labels)
    print(num_classes)
    name2num = {}
    num2name = {}
    for i in range(num_classes):
        l = labels[i]
        name2num[l] = i
        num2name[i] = l
    return (name2num, num2name)

def main():
    name2num, num2name = labeling('./food')
    f = open('label.txt', 'w+')
    count = 0
    for key, val in name2num.items():
        folder_dir = './food/' + key

        dir_prefix = '/food/' + key

        for img_name in os.listdir(folder_dir):
            img = cv2.imread(folder_dir+'/'+img_name, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            if img is None: continue
            os.rename(folder_dir+'/'+img_name, folder_dir+'/'+str(count) + '.jpg')
            f.write(dir_prefix+'/'+str(count) + '.jpg' + ' ' + str(val))
            f.write('\n')
            count += 1
    f.close()

main()


