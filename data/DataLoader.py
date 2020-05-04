import numpy as np
import argparse
import os
import pandas as pd
import random
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    '--input_file',
    type=str,
    default='./Digital_Music_only_rating.csv',
    help="the path towards the rating file to read data"
)
parser.add_argument(
    '-trd',
    '--train_dir',
    type=str,
    default='./train',
    help="the path for generated train csv files"
)
parser.add_argument(
    '-tsd',
    '--test_dir',
    type=str,
    default='./test',
    help="the path for generated test csv files"
)
parser.add_argument(
    '-nT',
    '--train_num',
    type=int,
    default=5,
    help="The number of interaction to be extracted as the training set, [-1 use leave 1 to test]"
)
opt = parser.parse_args()
print(opt)

if __name__ == "__main__":
    item_dict = {}
    user_dict = {}
    item_counter = 0
    user_counter = 0
    item_count_dict = {}
    user_count_dict = {}
    with open(opt.input_file, 'rt') as f_1:
        for cnt, line in enumerate(f_1):
            extract_data = line.strip().split(',')
            current_itemID = extract_data[0]
            current_userID = extract_data[1]
            current_rating = extract_data[2]
            
            if cnt % 20000 == 0:
                print("[Read Data]Reading {} lines".format(cnt))

            if current_itemID in item_dict:
                pass
            else:
                item_dict[current_itemID] = item_counter
                item_counter += 1
            
            if current_itemID in item_count_dict:
                item_count_dict[current_itemID] += 1
            else:
                item_count_dict[current_itemID] = 1

            if current_userID in user_dict:
                pass
            else:
                user_dict[current_userID] = user_counter
                user_counter += 1

            if current_userID in user_count_dict:
                user_count_dict[current_userID] += 1
            else:
                user_count_dict[current_userID] = 1

    print("Total items: {}".format(item_counter))
    print("Total users: {}".format(user_counter))

    # In order to project the 'index' to the real 'ID' for both user and item the dict needs to be writen in log files
    with open('./code_project/user_ID_index_mapping.csv', 'wt') as f_1:
        for key, value in item_dict.items():
            print(key + ',' + str(value), file=f_1)

    with open('./code_project/item_ID_index_mapping.csv', 'wt') as f_1:
        for key, value in user_dict.items():
            print(key + ',' + str(value), file=f_1)

    if opt.train_num == -1:
        # Use leave-one method to make testset
        # More specificly, if user only has 1 rating interaction, then this user won't be in the test
        # Else put one (user,item,rating) tuple into the test file
        with open(opt.input_file, 'rt') as f_1:
            with open(os.path.join(opt.train_dir, 'train_data.csv'), 'wt') as f_2:
                # First round, move all those user who only have 1 item into the trainset
                for cnt, line in enumerate(f_1):
                    extract_data = line.strip().split(',')
                    current_itemID = extract_data[0]
                    current_userID = extract_data[1]
                    current_rating = extract_data[2]
                    
                    if cnt % 20000 == 0:
                        print("[First Round]Reading {} lines".format(cnt))

                    if user_count_dict[current_userID] == 1:
                        # write this to a train set
                        print(str(user_dict[current_userID]) + ',' + 
                            str(item_dict[current_itemID]) + ',' + current_rating, file = f_2)
                        # print("In train: (user: {}, item: {}): rating: {}".format(user_dict[current_userID], item_dict[current_itemID], current_rating))
                        user_count_dict[current_userID] = 0

        with open(opt.input_file, 'rt') as f_1:
            with open(os.path.join(opt.train_dir, 'train_data.csv'), 'a') as f_2:
                with open(os.path.join(opt.test_dir, 'test_data.csv'), 'wt') as f_3:
                    # Second round, move all those user who have more than 1 items (v-1) items into trainset
                    for cnt, line in enumerate(f_1):
                        extract_data = line.strip().split(',')
                        current_itemID = extract_data[0]
                        current_userID = extract_data[1]
                        current_rating = extract_data[2]
                        
                        if cnt % 20000 == 0:
                            print("[Second]Reading {} lines".format(cnt))

                        if user_count_dict[current_userID] > 1:
                            # write this into a train set
                            print(str(user_dict[current_userID]) + ',' + 
                                    str(item_dict[current_itemID]) + ',' + current_rating, file = f_2)
                            user_count_dict[current_userID] -= 1
                            # print("In train: (user: {}, item: {}): rating: {}".format(user_dict[current_userID], item_dict[current_itemID], current_rating))
                        
                        elif user_count_dict[current_userID] == 1:
                            # write this into a test set
                            print(str(user_dict[current_userID]) + ',' + 
                                    str(item_dict[current_itemID]) + ',' + current_rating, file = f_3)
                            user_count_dict[current_userID] == 0
                            # print("In test: (user: {}, item: {}): rating: {}".format(user_dict[current_userID], item_dict[current_itemID], current_rating))

    elif opt.train_num > 1 and opt.train_num <= 5:
        # This means that for those user who only has less than opt.train_num interactions we will 
        # directly remove them from the dataset (both trainset and testset)
        # The method to recommend this user could be simly the Top-K most popular item
        count_filtered_user = 0
        count_filtered_item = 0
        with open(opt.input_file, 'rt') as f_1:
            # First round, move all those user who have less than opt.train_num items away from the datasets
            for cnt, line in enumerate(f_1):
                extract_data = line.strip().split(',')
                current_itemID = extract_data[0]
                current_userID = extract_data[1]
                current_rating = extract_data[2]
                
                if cnt % 20000 == 0:
                    print("[First Round]Reading {} lines".format(cnt))
                if user_count_dict[current_userID] < opt.train_num:
                    # set these userID's count to be 0
                    # print("In train: (user: {}, item: {}): rating: {}".format(user_dict[current_userID], item_dict[current_itemID], current_rating))
                    if user_count_dict[current_userID] != 0:
                        user_count_dict[current_userID] = 0
                        count_filtered_user += 1
                    else:
                        item_count_dict[current_itemID] -= 1
                        if item_count_dict[current_itemID] == 0:
                            count_filtered_item += 1

        with open(opt.input_file, 'rt') as f_1:
            with open(os.path.join(opt.train_dir, 'train_data_filter_'+str(opt.train_num)+'.csv'), 'wt') as f_2:
                with open(os.path.join(opt.test_dir, 'test_data_filter_'+str(opt.train_num)+'.csv'), 'wt') as f_3:
                    # Second round, move all those user who have more than opt.train_num items (|Interactions|-1) items into trainset
                    for cnt, line in enumerate(f_1):
                        extract_data = line.strip().split(',')
                        current_itemID = extract_data[0]
                        current_userID = extract_data[1]
                        current_rating = extract_data[2]
                        
                        if cnt % 20000 == 0:
                            print("[Second]Reading {} lines".format(cnt))

                        if user_count_dict[current_userID] > 1:
                            # write this into a train set
                            print(str(user_dict[current_userID]) + ',' + 
                                    str(item_dict[current_itemID]) + ',' + current_rating, file = f_2)
                            user_count_dict[current_userID] -= 1
                            # print("In train: (user: {}, item: {}): rating: {}".format(user_dict[current_userID], item_dict[current_itemID], current_rating))
                        
                        elif user_count_dict[current_userID] == 1:
                            # write this into a test set
                            print(str(user_dict[current_userID]) + ',' + 
                                    str(item_dict[current_itemID]) + ',' + current_rating, file = f_3)
                            user_count_dict[current_userID] == 0
                            # print("In test: (user: {}, item: {}): rating: {}".format(user_dict[current_userID], item_dict[current_itemID], current_rating))
        
        
        print("Total Filtered items: {}".format(count_filtered_item))
        print("Total Filtered users: {}".format(count_filtered_user))

    else:
        print('Leave for later implementation')
