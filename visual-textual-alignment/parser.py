import pickle
from collections import OrderedDict,defaultdict
from rdflib import URIRef, RDFS
import numpy as np

with open('.../dataset/dataset_1k.pkl', 'rb') as i:
    data = pickle.load(i)

def pre_process_data(data):

    model_data = defaultdict(list)

    #parse image data
    for img_id in data:
        img_data = data[img_id]
        #caption data
        caption_data = defaultdict(list)
        for cap_id,cap_data in img_data.items():
            #entites level data
            save_data = []
            for teid, entity_data in cap_data.items():
                visual_data = entity_data[0]
                textual_data = entity_data[1]

                #one textual entites are linked to more than one bounding box
                for i in visual_data:
                    save_data.append([[i[2][0][1000:],i[3]],[textual_data[1],textual_data[2]]])

            caption_data[cap_id]= save_data

        model_data[img_id] = caption_data

    return model_data

def all_possible_combination():
    dataset = pre_process_data(data)
    images_data = defaultdict(list)
    for img_id in dataset:
        img_data = dataset[img_id]
        caption_data = defaultdict(list)
        for cap_id, cap_data in img_data.items():
            possible_pairs_data = []
            for i in range(len(cap_data)):

                t = cap_data[i][1]
                for j in range(len(cap_data)):
                    v = cap_data[j][0]
                    if i ==  j:
                        possible_pairs_data.append([v,t,1])
                    else:
                        possible_pairs_data.append([v,t,0])

            caption_data[cap_id] = possible_pairs_data
        images_data [img_id] = caption_data

    return images_data

def training_test_set():
    dataset = all_possible_combination()

    train_data = defaultdict(list)
    test_data = defaultdict(list)

    count = 0
    for img_id in dataset:
        img_data = dataset[img_id]
        captions_data = defaultdict(list)

        if count<800:
            for cap_id,cap_data in img_data.items():
                true_pair = 0
                pair_data = []
                for i in range(len(cap_data)):
                    if cap_data[i][2]==1:
                        pair_data.append(cap_data[i])
                        true_pair+=1

                for i in range(len(cap_data)):
                    if cap_data[i][2] !=1:
                        true_pair-=1
                        pair_data.append(cap_data[i])

                        if true_pair ==0:
                            break
                captions_data[cap_id] = pair_data
            train_data [img_id] = captions_data
        else:
            test_data[img_id] = img_data
        count+=1

    return train_data,test_data

train_data,test_data = training_test_set()

with open(".../dataset/training_data.pkl","wb") as outfile:
    pickle.dump(train_data,outfile)

with open(".../dataset/test_data.pkl","wb") as outfile:
    pickle.dump(test_data,outfile)
