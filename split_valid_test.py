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
    samples_num = 105*len(_people)
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
        
        img1_index = random.randint(0, 14)
        img2_index = random.randint(0, 14)
          
        _set.append([file1, img1[img1_index] , file2, img2[img2_index], 0,1])

    random.shuffle(_set)
    return _set

def build_dataset(src_folder):
    test_people, valid_people = [], []
    
    people_folder = read_csv_file(src_folder)
    random.shuffle(people_folder)
    valid_people = people_folder[0:1800]
    test_people = people_folder[1800:3600]

    valid_set = get_pair(valid_people)
    test_set = get_pair(test_people)

    random.shuffle(valid_set)
    random.shuffle(test_set)

    print('\tpeople\tpicture')
    print('valid:\t%6d\t%7d' % (len(valid_people), len(valid_set)))
    print('test:\t%6d\t%7d' % (len(test_people), len(test_set)))
    return valid_set, test_set

if __name__ == '__main__':
    src_file    = "C:/Users/bear/cropwebface/index_15_1.csv"
    valid_set_file  = "C:/Users/bear/cropwebface/valid_set_15.csv"
    test_set_file = "C:/Users/bear/cropwebface/test_set_15.csv"
     
    valid_set, test_set = build_dataset(src_file)
    set_to_csv_file(valid_set,  valid_set_file)
    set_to_csv_file(test_set, test_set_file)
