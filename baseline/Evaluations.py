import pickle

#------------ upload files for 1k documents ---------------#
with open(".../dataset/vtkel_pairs.pkl","rb") as infile:
    vtkel_pairs_1k = pickle.load(infile)

with open(".../dataset/linker_pairs_1k.pkl","rb") as infile:
    linker_pairs_resnet_1k = pickle.load(infile)

#------------ upload files for 31k documents ---------------#
with open(".../dataset/vtkel_pairs_31k.pkl","rb") as infile:
    vtkel_pairs_31k = pickle.load(infile)

with open(".../dataset/linker_pairs_31k.pkl","rb") as infile:
    linker_pairs_resnet_31k = pickle.load(infile)

#Total pairs in Benchmark (VTKEL)
def Benchmark(vtkel_pairs):
    benchmark_pairs_count = 0

    for img_id in vtkel_pairs:
        img_data_vtkel = vtkel_pairs[img_id]
        for i in img_data_vtkel:
            benchmark_pairs_count+=1

    return benchmark_pairs_count

#Total pairs predicted by VT-LinKEr
def VT_LinKEr_pairs(linker_pairs_data):
    linker_pairs_count = 0

    for img_id in linker_pairs_data:
        img_data_linker = linker_pairs_data[img_id]
        for i in img_data_linker:
            linker_pairs_count+=1

    return linker_pairs_count

#--------------Evaluation part --------------------------------
def intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
    return iou

def LinKer_Evaluation(confidence,vtkel_pairs,linker_pairs):
    img_count = 0
    match_count = 0
    for img_id in linker_pairs:
        img_data_linker = linker_pairs[img_id]
        img_data_vtkel = vtkel_pairs[img_id]
        #visual-part
        for i in img_data_linker:
            count = 0
            linker_bb = i[1]
            linker_te = i[2]

            for j in img_data_vtkel:
                count+=1
                vtkel_bb = j[0]
                vtkel_te = j[1]

                iou = intersection_over_union(linker_bb,vtkel_bb)
                if iou>=confidence and linker_te==vtkel_te:
                    # print(iou, linker_te[37:],vtkel_te[37:])
                    match_count+=1

        img_count+=1

    return match_count

#-------Precesion and Recall---------
def results(threshold,dataset):

    if dataset == "1k":
        benchmark_pairs_count = Benchmark(vtkel_pairs_1k)
        linker_pairs_count = VT_LinKEr_pairs(linker_pairs_resnet_1k)
        match_count = LinKer_Evaluation(threshold,vtkel_pairs_1k,linker_pairs_resnet_1k)

    elif dataset == "31k":
        benchmark_pairs_count = Benchmark(vtkel_pairs_31k)
        linker_pairs_count = VT_LinKEr_pairs(linker_pairs_resnet_31k)
        match_count = LinKer_Evaluation(threshold,vtkel_pairs_31k,linker_pairs_resnet_31k)

    P = (match_count/linker_pairs_count)*100
    R = (match_count/(benchmark_pairs_count))*100
    F1 = 2*((P*R)/(P+R))
    print(match_count,linker_pairs_count,benchmark_pairs_count)
    print('Precesion:',P)
    print('Recall:',R,'\n')
    # print('F1:',F1)

results(threshold=0.5,dataset="1k")
# results(threshold=0.5, "31k")
