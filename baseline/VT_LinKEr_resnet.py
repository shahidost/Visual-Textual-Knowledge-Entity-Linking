import cv2
from collections import defaultdict
import numpy as np
import pickle
from rdflib import URIRef, RDFS, Graph, ConjunctiveGraph
from collections import Counter

#------------upload files for 1k VTKEL dataset----------------#
with open(".../dataset/resent_data_1k.pkl","rb") as infile:
    resent_data_1k = pickle.load(infile)

with open(".../dataset/vktel_textual_entities_1k.pkl","rb") as infile:
    textual_data_1k = pickle.load(infile)

#------------upload files for 31k VTKEL dataset----------------#
with open(".../dataset/resent_data_31k.pkl","rb") as infile:
    resent_data_31k = pickle.load(infile)

with open(".../dataset/vktel_textual_entities_31k.pkl","rb") as infile:
    textual_data_31k = pickle.load(infile)

with open(".../dataset/resnet_subclasses.pkl","rb") as infile:
    resnet_subclasses = pickle.load(infile)

with open(".../dataset/resnet_supclasses.pkl","rb") as infile:
    resnet_supclasses = pickle.load(infile)

def visual_textual_alignment(dataset):

    if dataset == "1k":
        visual_data = resent_data_1k
        textual_data = textual_data_1k
    elif dataset == "31k":
        visual_data = resent_data_31k
        textual_data = textual_data_31k

    linker_pairs = defaultdict(list)
    count = 0

    for img_id in textual_data:
        img_visual_data = visual_data[str(img_id)]
        img_textual_data = textual_data[img_id]

        linker_pairs_for_img = []
        count+=1

        for te_id, te_data in img_textual_data.items():
            #textual entity type
            te_type = te_data[0]

            #alignment with all possible combination
            for i in range(len(img_visual_data[0])):
                ve_type = img_visual_data[0][i]
                # threshold = img_visual_data[1][i]
                bb = img_visual_data[2][i]

                if (te_type in resnet_subclasses[ve_type] or te_type in resnet_supclasses[ve_type]):
                    linker_pairs_for_img.append([te_type,bb,te_id])

        linker_pairs[img_id] = linker_pairs_for_img
        print(count,img_id)

    return linker_pairs

linker_pairs = visual_textual_alignment("1k")
with open(".../dataset/linker_pairs_1k.pkl","wb") as outfile:
    pickle.dump(linker_pairs,outfile)

# linker_pairs = visual_textual_alignment("31k")
# with open(".../dataset/linker_pairs_31k.pkl","wb") as outfile:
    # pickle.dump(linker_pairs,outfile)