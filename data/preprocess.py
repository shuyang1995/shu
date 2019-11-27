import os

path = '/home/boyuan/foodRecognition/data/food/baixia'

fileList = os.listdir(path)




for idx in range(len(fileList)):
  fileName = path+'/'+fileList[idx]
  os.system("mv "+fileName+' ' + path + '/'+str(idx) + '.jpg')



