import dlib
import numpy
import csv
from PIL import Image
import sys
import os
import random

def read_csv_file(csv_file):
    with open(csv_file,'r') as csvfile:
       reader = [each for each in csv.reader(csvfile)]
    return reader

def set_to_csv_file(data_set, file_name):
    with open(file_name, "w") as f:
        for item in data_set:
            print(" ".join(map(str, item)), file=f)
    f.close()

def get_landmarks(im):
    rects = detector(im, 1)
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def img_seg(im, location, length):
   dx = length
   dy = length
   if (location[1] - dx) < 0:
        dx = location[1]

   if  (location[1] + dx) > im.shape[0]:
        dx = im.shape[0] - location[1]

   if (location[0] - dy) < 0:
        dy = location[0]

   if  (location[0] + dy) > im.shape[1]:
        dy = im.shape[1] - location[0]
   im = im[(location[1] - dx) : (location[1] + dx) , (location[0] - dy) : (location[0]+dy)]
   im = Image.fromarray(numpy.uint8(im))
   im = im.resize((31,31))
   return im

#源程序是用sys.argv从命令行参数去获取训练模型，精简版我直接把路径写在程序中了
predictor_path = "./data/shape_predictor_68_face_landmarks.dat"

#源程序是用sys.argv从命令行参数去获取文件夹路径，再处理文件夹里的所有图片
#这里我直接把图片路径写在程序里了，每运行一次就只提取一张图片的关键点
faces_path = "C:/webface/"
result_folder = 'C:/Users/bear/cropwebface'

people_imgs = []
if not os.path.exists(result_folder):
        os.mkdir(result_folder)
people_count = 0
people_folders = read_csv_file('C:/Users/bear/cropwebface/index_27_1.csv')
#people_folders = people_folders[1210:]
face_number = 1
for people_folder in people_folders:
    file_img = people_folder[0].strip().split()
    
    src_people_path = os.path.join(faces_path, file_img[0])
    dest_people_path = os.path.join(result_folder, file_img[0])
    img_path = os.listdir(src_people_path)
    imgs = file_img[1:]
    for old_img in imgs:
        img_path.remove(old_img)
    index_max = len(img_path)
    
    if index_max < face_number:
        continue
    random.shuffle(img_path)
    face_count = 0
    index = 0
    people_img_path = file_img
    if not os.path.exists(dest_people_path):
        os.mkdir(dest_people_path)
    while  face_count < face_number and (index_max - index) >= (face_number - face_count):
        img_file = img_path[index]
        index = index + 1
        #for img_file in os.listdir(src_people_path):
        src_img_path = os.path.join(src_people_path, img_file)
        #与人脸检测相同，使用dlib自带的frontal_face_detector作为人脸检测器
        img = Image.open(src_img_path)
        img = numpy.asarray(img)
        img.flags.writeable = True
        
        detector = dlib.get_frontal_face_detector()
        dets = detector(img, 1)
        # 检测到一张人脸
        if (len(dets) == 1):
            #使用官方提供的模型构建特征提取器
            predictor = dlib.shape_predictor(predictor_path)
            #get_landmarks()函数会将一个图像转化成numpy数组，并返回一个68 x2元素矩阵，输入图像的每个特征点对应每行的一个x，y坐标。
            landmarks = get_landmarks(img)
            left_mouth_location   =  numpy.ravel(landmarks[48,:])
            right_mouth_location =  numpy.ravel(landmarks[54,:])
            nose_tip_location       =  numpy.ravel(landmarks[30,:])
            left_eye_location =   numpy.mean(landmarks[36:42], axis=0)
            left_eye_location = left_eye_location.astype(int)
            left_eye_location = numpy.ravel(left_eye_location)
            right_eye_location = numpy.mean(landmarks[42:48], axis=0)
            right_eye_location = right_eye_location.astype(int)
            right_eye_location = numpy.ravel(right_eye_location)

            location = numpy.vstack((left_mouth_location,right_mouth_location,nose_tip_location,left_eye_location,right_eye_location))
            
            if  (not numpy.all(location > 0)) or (not numpy.all((location[:,1] - img.shape[0]) < 0)) or (not numpy.all((location[:,0]  -  img.shape[1]) < 0)):
                continue
            
            #people_img_path.append(people_folder)
            people_img_path.append(img_file)
    
            # 4
            left_mouth_length = landmarks[48,1] - min([landmarks[40,1],landmarks[41,1],landmarks[46,1],landmarks[47,1]])
            left_mouth_length =  left_mouth_length * 5 // 4
            img_lm_l =  img_seg(img,left_mouth_location,left_mouth_length)
           #4
            right_mouth_length = landmarks[54,1] - min([landmarks[40,1],landmarks[41,1],landmarks[46,1],landmarks[47,1]])
            right_mouth_length = right_mouth_length * 5 // 4
            img_rm_l =  img_seg(img,right_mouth_location,right_mouth_length)
            #4
            nose_tip_length = landmarks[30,1] - min([landmarks[19,1],landmarks[24,1]])
            nose_tip_length = nose_tip_length * 5 // 4
            img_nose_l = img_seg(img,nose_tip_location,nose_tip_length)
            #left eye
            #3
            left_eye_length  = landmarks[33,1] - left_eye_location[1]
            left_eye_length  = left_eye_length * 5 // 3
            img_le_l = img_seg(img,left_eye_location,left_eye_length)
            #right eye
             #3
            right_eye_length  = landmarks[33,1] - right_eye_location[1]
            right_eye_length  = right_eye_length * 5 // 3
            img_re_l = img_seg(img,right_eye_location,right_eye_length)
  
            dest_img_path =  dest_people_path +  '//rgb_' +  'large_' + 'left_eye_' + img_file
            img_le_l.save(dest_img_path)
            dest_img_path =  dest_people_path +  '//rgb_' +  'large_' + 'right_eye_' + img_file
            img_re_l.save(dest_img_path)
            dest_img_path =  dest_people_path +  '//rgb_' +  'large_' + 'nose_point_' + img_file
            img_nose_l.save(dest_img_path)
            dest_img_path =  dest_people_path +  '//rgb_' +  'large_' + 'left_mouth_' + img_file
            img_lm_l.save(dest_img_path)     
            dest_img_path =  dest_people_path +  '//rgb_' +  'large_' + 'right_mouth_' + img_file
            img_rm_l.save(dest_img_path)   
            face_count += 1
            
    people_count += 1
    if face_count == face_number:
        people_imgs.append(people_img_path)
    if people_count % 100 == 0:
        print ('processing' , people_count, 'person' )
        set_to_csv_file(people_imgs, 'C:/Users/bear/cropwebface/index_backup.csv')

set_to_csv_file(people_imgs, 'C:/Users/bear/cropwebface/index_28.csv')



