import sys, os, random
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd

import cv2

import matplotlib
import matplotlib.pyplot as plt 

# Helper   
    
def get_sample_weigth(data, bins=90, minimum=1, maximum=10):
    """compute sample weigth according to the bins number
    Args:
	data: dataframe containing informations concerning image aesthetics ration
    bins: number of bins
    minimum: low level of the mean rating admited
    maximum: upper level of the mean rating admited
    """
    card  = data.shape[0]
    subsets = [minimum]
    weigths = []
    subu = minimum
    for i in range(1,bins+1):
        subl = subu
        subu = (maximum-minimum)*i/bins + minimum 
        cardx = data[(data["score_mean"]>=subl) & (data["score_mean"]<subu)].shape[0]
        cardn = cardx/card
        weigth = 1/cardn if cardn != 0 else np.NAN
        subsets.append(subu)
        weigths.append(weigth)
        
    assert len(subsets)==bins+1
    assert len(weigths)==bins
    
    def sample_weigth(x):
        for i in range(1,bins+1):
            if x>=subsets[i-1] and x < subsets[i]:
                return weigths[i-1]
        return np.NAN
    data["sample_weigth"] = data["score_mean"].apply(lambda x: sample_weigth(x))
    assert data.shape[0] == data["sample_weigth"].dropna().shape[0]
    return subsets, weigths


def load_images_metadata(path, image_folder="pre_images"):
    """Load images aesthetic ratin data, compute the mean, the standard deviation and the mode of rating distribution
    Args:
	path: directory containing the folder containing all files related to aesthetic rating
    """
    images_files = [f for f in listdir(path+image_folder) if isfile(join(path+image_folder, f))]
    images_ids = [int(f[0:-4]) for f in images_files if  f[-4:].lower() == '.npy']
    
    aesthetic_levels =["level "+str(i) for i in range(1,11)]
    columns_names = ["N°","id"]+aesthetic_levels+["Semantic tag ID1","Semantic tag ID2","Challenge ID"]
    aesthetics_rating = pd.read_csv(path+"metadata/AVA.txt",sep=" ",names=columns_names)
    # Chech if any missing image and remove their metadata
    missing_images_ids = set(aesthetics_rating.id.tolist()) - set(images_ids)
    aesthetics_rating = aesthetics_rating[False == aesthetics_rating.id.isin(list(missing_images_ids))]
    
    # Normalise the aeshetic rating
    print("start preprocessing...", aesthetics_rating.shape)
    aesthetics_rating[aesthetics_rating.columns.tolist()[2:12]] = aesthetics_rating[aesthetics_rating.
                                                                                    columns.tolist()[2:12]].apply(lambda x: x/np.sum(x),axis=1)
    print("mean estimation of image aesthetic rating...")
    aesthetics_rating["score_mean"] = aesthetics_rating[aesthetics_rating.columns.tolist()[2:12]
                                                              ].apply(lambda x: np.sum(x*np.arange(1,11)),
                                                                      axis=1)
    print("standard estimation of image aesthetic rating...")
    aesthetics_rating["score_std"] = aesthetics_rating[aesthetics_rating.columns.tolist()[2:12]+["score_mean"]
                                                              ].apply(lambda x: np.sqrt(np.sum(x[:10]*(np.arange(1,11)-x[10])**2)),
                                                                      axis=1)
    print("mode estimation of image aesthetic rating...")
    aesthetics_rating["score_peak"] = aesthetics_rating[aesthetics_rating.columns.tolist()[2:12]
                                                              ].apply(lambda x: int(np.argmax(x)[5:]),
                                                                      axis=1)
    print("image cluster label estimation based on image mean rating...")
    aesthetics_rating["cluster"] = aesthetics_rating["score_mean"].apply(lambda x:get_cluster_number(x))
    
    print("sample weigth estimation...")
    get_sample_weigth(aesthetics_rating)
    
    assert aesthetics_rating.shape[0] == len(images_ids)
    return aesthetics_rating

def get_train_test(data):
    """Split data into one training, one validation following data distribution and two test one following data distribution and another following uniform distribution
    Args:
	data: images aesthetic preprocessed data
    """
    ids = data.id.values
    RD_ids = random.sample(set(ids),5000)
    
    remainder_ids = set(ids) - set(RD_ids)
    
    #
    remainder = data[data.id.isin(list(remainder_ids))]
    low_ids = remainder[remainder["score_mean"] < 4].id.values
    medium_ids = remainder[(remainder["score_mean"] >= 4) & (remainder["score_mean"] <=7)].id.values
    high_ids = remainder[remainder["score_mean"] > 7].id.values
    
    ED_ids_low = random.sample(set(low_ids),1000)
    ED_ids_med = random.sample(set(medium_ids),1000)
    ED_ids_high = random.sample(set(high_ids),1000)
    
    ED_ids = ED_ids_low + ED_ids_med + ED_ids_high
    
    remainder_ids = set(remainder_ids) - set(ED_ids)

    n_val = int(0.2*len(remainder_ids))
    val_ids = random.sample(set(remainder_ids),n_val)

    train_ids = set(remainder_ids) - set(val_ids)
    
    train = data[data.id.isin(list(train_ids))]
    val = data[data.id.isin(list(val_ids))]
    RD_test = data[data.id.isin(list(RD_ids))]
    ED_test = data[data.id.isin(list(ED_ids))]

    return train, val, RD_test, ED_test

def get_cluster_number(x, bins=20, minimum=1, maximum=10):
    """Computer the cluster number based on mean rating score x of the image
    Args:
	x: mean score
    bins: cluster number
    minimum: mean score low bound
    maximum: mean score upper bound
    """
    for i in range(1,bins+1):
        if x < (maximum-minimum)*i/bins + minimum:
            return (maximum-minimum)*(i-0.5)/bins + minimum

def get_train_clusters(data, min_len):
    """group images ids according to their cluster number
    Args:
	data: images aesthetics rating
    """
    labels = list(np.unique(data["cluster"].values))
    clusters = {label:data[data["cluster"]==label].id.values.tolist() for label in labels if len(data[data["cluster"]==label].id.values.tolist()) >= min_len}
    return clusters


def get_mini_bacth(cluster, size):
    """Yield successive mini_batches from cluster.
    Args:
	cluster: list of images ids belonging to the same cluster
        size: number of image to sample per each cluster
    """
    random.shuffle(cluster)
    for i in range(0, len(cluster), size):
        if i+size > len(cluster):
            yield list(cluster[i:i + size]) + list(random.sample(set(cluster),i+size-len(cluster)))
        else:
            yield cluster[i:i + size]
        
def get_batch(mini_batches, mini_size, clusters):
    """
	get a batch where images are equally sampled from each cluster
	@param cluster_mini_batches: bacthes of each cluster
	@param mini_size: the size of the mini batch
	@cluster: cluster of images according to their mean rating
    """
    batch = []
    for i in clusters.keys():
        try:
            batch += list(next(mini_batches[i]))
        except: 
            mini_batches[i] = get_mini_bacth(clusters[i], mini_size)
            batch += list(next(mini_batches[i]))
    return batch

def load_images_labels(batch, data, path, image_size=224, image_folder="pre_images/", 
                       model_type="Vgg16", estimation_type="mean", data_biais="balancing"):
    """Load batch images and labels
    Args:
        batch : list of batch images ids
        data : images aesthetics rating
        path : folder containing image folder
    """
    b_size = len(batch)
    images = np.empty((b_size,image_size,image_size,3))
    
    if data_biais == "weigthed":
        if  estimation_type=="mean":
            labels = np.empty((b_size,2))
            target = ["score_mean","sample_weigth"]
        else:
            labels = np.empty((b_size,11))
            target = ["level "+str(i) for i in range(1,11)] + ["sample_weigth"]
    else:
        if  estimation_type=="mean":
            labels = np.empty((b_size,1))
            target = "score_mean"
        else:
            labels = np.empty((b_size,10))
            target = ["level "+str(i) for i in range(1,11)]
        
    for i in range(b_size):
        images[i] = np.load(path+image_folder+ str(batch[i])+".npy")
        labels[i] = data[data.id==batch[i]][target].values[0]
    if model_type != "Vgg16" and np.max(images) != 1.0:
        images = (images/255.0 -0.5)*2
    return images, labels

def load_images_labels2(batch, data, path, image_size=224, image_folder="Images/", model_type="mobilenet"):
    """Load batch images and labels
    Args:
        batch : list of batch images ids
        data : images aesthetics rating
        path : folder containing image folder
    """
    b_size = len(batch)
    target = ["level "+str(i) for i in range(1,11)]
    images = np.empty((b_size,image_size,image_size,3))
    labels = np.empty((b_size,10))

    if model_type=="Vgg16":
        VGG_MEAN = [103.939, 116.779, 123.68]

        for i in range(b_size):
            images[i] = cv2.resize(cv2.imread(path+image_folder+ str(batch[i])+".jpg"),(image_size, image_size))
            labels[i] = data[data.id==batch[i]][target].values[0]
        images[:,:,:,0] -= VGG_MEAN[0]
        images[:,:,:,1] -= VGG_MEAN[1]
        images[:,:,:,2] -= VGG_MEAN[2]
    else:
        for i in range(b_size):
            images[i] = cv2.resize(cv2.imread(path+image_folder+ str(batch[i])+".jpg")[...,::-1],(image_size, image_size))
            labels[i] = data[data.id==batch[i]][target].values[0]
        images = (images/255.0 -0.5)*2

    return images, labels

def histogram_of_image(idx, data, path):
    """display the rating information of image
    Args:
	idx: image id
	data: images aesthetic data file
	@path: directory contining images folder
    """
    aesth_levels = data.columns.tolist()[2:12]
    y_pos = np.arange(len(aesth_levels))
    rating = data[data.id==idx][aesth_levels].values[0]
    plt.bar(y_pos, rating, align='center', alpha=0.5)
    plt.xticks(y_pos, aesth_levels)
    plt.ylabel('rating')
    plt.title('Aesthetic level')
    plt.show()
    
    img = cv2.imread(path+"Images/"+str(idx)+".jpg")
    plt.imshow(img)
    plt.show()
