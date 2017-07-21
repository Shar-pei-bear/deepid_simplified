import os
import os.path
import random
import csv
from itertools import combinations
def set_to_csv_file(data_set, file_name):
    with open(file_name, "w") as f:
        for item in data_set:
            print(" ".join(map(str, item)), file=f)

def read_csv_file(csv_file):
    with open(csv_file,'r') as csvfile:
       reader = [each for each in csv.reader(csvfile)]
    return reader

def get_pair(_people):
    _set= []
    samples_num = 351*len(_people)
    for i, people_img in enumerate(_people):
         file_img = people_img[0].strip().split()
         file = file_img[0]
         img = file_img[1:]
         positive_samples =  list(combinations(img, 2))
         for positive_sample in positive_samples:
              _set.append([file, positive_sample[0], file, positive_sample[1],1,0])
              
    negative_samples =  list(combinations(_people, 2))
    random.shuffle(negative_samples)
    negative_samples = negative_samples[0:samples_num]
    
    for negative_sample in negative_samples:
        sample1 = negative_sample[0]
        sample2 = negative_sample[1]
        
        file_img1   = sample1[0].strip().split()
        file_img2   = sample2[0].strip().split()

        file1 = file_img1[0]
        file2 = file_img2[0]

        img1= file_img1[1:]
        img2= file_img2[1:]
        
        img1_index = random.randint(0, 26)
        img2_index = random.randint(0, 26)
          
        _set.append([file1, img1[img1_index] , file2, img2[img2_index], 0,1])

    random.shuffle(_set)
    return _set

def build_dataset(src_folder):
    RBM_people, train_people = [], []
    
    people_folder = read_csv_file(src_folder)
    random.shuffle(people_folder)
    train_people = people_folder[0:3000]
    RBM_people = people_folder[3000:3750]

    train_set = get_pair(train_people)
    RBM_set = get_pair(RBM_people)

    random.shuffle(RBM_set)
    random.shuffle(train_set)

    print('\tpeople\tpicture')
    print('RBM:\t%6d\t%7d' % (len(RBM_people), len(RBM_set)))
    print('train:\t%6d\t%7d' % (len(train_people), len(train_set)))
    return RBM_set, train_set

if __name__ == '__main__':
    src_file    = "C:/Users/bear/cropwebface/index_27.csv"
    RBM_set_file  = "C:/Users/bear/cropwebface/RBM_set_27.csv"
    train_set_file = "C:/Users/bear/cropwebface/train_set_27.csv"
     
    RBM_set, train_set = build_dataset(src_file)
    set_to_csv_file(RBM_set,  RBM_set_file)
    set_to_csv_file(train_set, train_set_file)
