import numpy as np
import argparse
import os
import pandas as pd
import random
from collections import defaultdict
import json

##########################################################
# This file is used to generated the train and test data #
# from the original raw data of rating records.          #
##########################################################

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
    default=-1,
    help="The number of interaction to be extracted as the training set, [-1 use leave 1 to test]"
)
opt = parser.parse_args()
print(opt)

if __name__ == "__main__":
    num_neg_pos_tuple = 99    # number of 'negative' data(tuple) which need to select from the non-rating region.
    item_dict = {}
    user_dict = {}
    item_counter = 0
    user_counter = 0
    item_count_dict = {}
    user_count_dict = {}
    user_pos_rating_dict = {} # record the number of the positive rating records for this user
    user_pos_rating_set = defaultdict(set)     # record all the positive item of a user

    ## NOTE:[UPDATE] This is used to generate the trained data from the whole dataset ##
    with open(opt.input_file, 'rt') as f_1:
        for cnt, line in enumerate(f_1):
            extract_data = line.strip().split(',')
            current_itemID = extract_data[0]
            current_userID = extract_data[1]
            current_rating = float(extract_data[2])
            
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

            if current_rating > 2.1:
                if current_userID in user_pos_rating_dict:
                    user_pos_rating_dict[current_userID] += 1
                else:
                    user_pos_rating_dict[current_userID] = 1
                user_pos_rating_set[current_userID].add(item_dict[current_itemID])

    print("Total items: {}".format(item_counter))
    print("Total users: {}".format(user_counter))

    # In order to project the 'index' to the real 'ID' for both user and item the dict needs to be writen in log files
    with open('./code_project/user_ID_index_mapping_init.csv', 'wt') as f_1:
        for key, value in item_dict.items():
            print(key + ',' + str(value), file=f_1)

    with open('./code_project/item_ID_index_mapping_init.csv', 'wt') as f_1:
        for key, value in user_dict.items():
            print(key + ',' + str(value), file=f_1)

    ## NOTE:[UPDATE] Using the json file as dataset which is the 5-core data.

    if opt.train_num == -1:
        # Use leave-one method to make testset
        # More specificly, if user only has 1 rating interaction, then this user won't be in the test
        # Else put one (user,item,rating) tuple into the test file
        #################################################################################################################
        # NOTE: [UPDATE] To make this more reasonable, if we don't have a positive data in the training set, then we    #
        # don't add any interaction of this user to the test set. If we have exactly 1 positive data in the training    #
        # set, we add this data to the final test file which contains 1 positive data(tuple) which belongs to this user #
        # with other 99 randomly selected negative data(tuple) which didn't occured in the interaction(rating) matrix.  #
        # If we have more than 1 (>=2) positive rating data(tuple) for this specific user, we take 1 from them and add  #
        # it as the positve test data(tuple) and then randomly select 99 negative data which didn't occurred in the     #
        # training set of this specific user.                                                                           #
        #################################################################################################################
        with open(opt.input_file, 'rt') as f_1:
            with open(os.path.join(opt.train_dir, 'train_data.csv'), 'wt') as f_2:
                with open(os.path.join(opt.test_dir, 'test_data.csv'), 'wt') as f_3:
                    with open(os.path.join(opt.test_dir, 'test_data_with_neg.csv'), 'wt') as f_4:
                        # First round, move all those user who only have 1 item into the trainset
                        # NOTE: [UPDATE] check if the rating here is positive, if it is positive then we 
                        # generate the corresponding test set.
                        # NOTE: [UPDATE] this is not reasonable...
                        for cnt, line in enumerate(f_1):
                            extract_data = line.strip().split(',')
                            current_itemID = extract_data[0]
                            current_userID = extract_data[1]
                            current_rating = float(extract_data[2])
                            
                            if cnt % 20000 == 0:
                                print("[First Round]Reading {} lines".format(cnt))

                            if user_count_dict[current_userID] == 1:
                                # write this to a train set
                                print(str(user_dict[current_userID]) + ',' + 
                                        str(item_dict[current_itemID]) + ',' + str(current_rating), file = f_2)
                                # print("In train: (user: {}, item: {}): rating: {}".format(user_dict[current_userID], item_dict[current_itemID], current_rating))
                                user_count_dict[current_userID] = 0
                                # if current_rating > 2.1 and (current_userID in user_pos_rating_dict):
                                #     # save it in the test_data.csv
                                #     print(str(user_dict[current_userID]) + ',' + 
                                #             str(item_dict[current_itemID]) + ',' + str(current_rating), file = f_3)
                                #     # save it in the test_data_with_neg.csv
                                #     ## First try to find some plain interaction data (non-rating) of this user. Number of these tuples: num_neg_pos_tuple
                                #     generated_neg = set()
                                #     for i in range(num_neg_pos_tuple):
                                #         while True:
                                #             generate_item = np.random.randint(0, item_counter)
                                #             if generate_item in generated_neg:
                                #                 continue
                                #             elif generate_item in user_pos_rating_set[current_userID]:
                                #                 continue
                                #             else:
                                #                 generated_neg.add(generate_item)
                                #                 break
                                #     # write these generated
                                #     string_test_chain = ''
                                #     string_test_chain += str(user_dict[current_userID]) + ',' + str(item_dict[current_itemID]) + ','
                                #     for item in generated_neg:
                                #         string_test_chain += str(item)
                                #         string_test_chain += ','
                                #     string_test_chain = string_test_chain.strip(',')
                                #     print(string_test_chain, file = f_4)


        with open(opt.input_file, 'rt') as f_1:
            with open(os.path.join(opt.train_dir, 'train_data.csv'), 'a') as f_2:
                with open(os.path.join(opt.test_dir, 'test_data.csv'), 'a') as f_3:
                    with open(os.path.join(opt.test_dir, 'test_data_with_neg.csv'), 'a') as f_4:
                        # Second round, move all those user who have more than 1 items (v-1) items into trainset
                        # NOTE: [UPDATE] When adding a data(tuple) into the testset, we not only the (user, item, rating)
                        # tuple into the test.csv file but also add several negative items into the test_data_with_neg.csv
                        # So that we can generate a test set with some 'negative'(which we don't have interaction on them)
                        # items. Number of these item: num_neg_pos_tuple
                        for cnt, line in enumerate(f_1):
                            extract_data = line.strip().split(',')
                            current_itemID = extract_data[0]
                            current_userID = extract_data[1]
                            current_rating = float(extract_data[2])
                            
                            if cnt % 20000 == 0:
                                print("[Second]Reading {} lines".format(cnt))

                            if user_count_dict[current_userID] > 1:
                                # write this into a train set
                                print(str(user_dict[current_userID]) + ',' + 
                                        str(item_dict[current_itemID]) + ',' + str(current_rating), file = f_2)
                                user_count_dict[current_userID] -= 1
                                # print("In train: (user: {}, item: {}): rating: {}".format(user_dict[current_userID], item_dict[current_itemID], current_rating))
                            
                            elif user_count_dict[current_userID] == 1:
                                # write this into a test set
                                print(str(user_dict[current_userID]) + ',' + 
                                        str(item_dict[current_itemID]) + ',' + str(current_rating), file = f_3)
                                user_count_dict[current_userID] == 0
                                # print("In test: (user: {}, item: {}): rating: {}".format(user_dict[current_userID], item_dict[current_itemID], current_rating))
                                # Generate negative items into test
                                generated_neg = set()
                                for i in range(num_neg_pos_tuple):
                                    while True:
                                        generate_item = np.random.randint(0, item_counter)
                                        if generate_item in generated_neg:
                                            continue
                                        elif generate_item in user_pos_rating_set[current_userID]:
                                            continue
                                        else:
                                            generated_neg.add(generate_item)
                                            break
                                # write there generated item into the file. merge them as one string
                                string_test_chain = ''
                                string_test_chain += str(user_dict[current_userID]) + ',' + str(item_dict[current_itemID]) + ','
                                for item in generated_neg:
                                    string_test_chain += str(item)
                                    string_test_chain += ','
                                string_test_chain = string_test_chain.strip(',')
                                print(string_test_chain, file = f_4)

    elif opt.train_num > 1 and opt.train_num <= 5:
        # This means that for those user who only has less than opt.train_num interactions we will 
        # directly remove them from the dataset (both trainset and testset)
        # The method to recommend this user could be simly the Top-K most popular item
        ########
        # NOTE: [UPDATE ISSUE] there should be similar way to generate the test set with negative examples 
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
