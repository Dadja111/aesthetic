import sys
sys.path.insert(0,"../utils")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] =  sys.argv[1]

print('CUDA_VISIBLE_DEVICES',os.environ["CUDA_VISIBLE_DEVICES"])

import helper as hp
import random

import tensorflow as tf
path_prefix = "../../../../"
path = path_prefix+"../dataset/AVA/"

test_type = sys.argv[2]
data_bias = sys.argv[3]
cost_type = sys.argv[4] #python eval_mob.py 2 E balancing kl
if "E" == test_type:
    test = hp.pd.read_csv(path_prefix+"data/ED_test.csv")
elif "R" == test_type:
    test = hp.pd.read_csv(path_prefix+"data/RD_test.csv")
elif "V" == test_type:
    test = hp.pd.read_csv(path_prefix+"data/val.csv")
batch_size = 300
print(batch_size)

tf.reset_default_graph()
with tf.Session() as sess:
    step = 0
    saver = tf.train.import_meta_graph("../../outputs/ckpts/final/"+data_bias+"/aesthetic_"+cost_type+".ckpt.meta")
    saver.restore(sess, "../../outputs/ckpts/final/"+data_bias+"/aesthetic_"+cost_type+".ckpt") #aesthetic_kl.ckpt
    graph = tf.get_default_graph()
    inputs = graph.get_tensor_by_name("input:0")
    print(inputs.get_shape().as_list())
    kp = graph.get_tensor_by_name("keep_probability:0")
    distrib = tf.nn.softmax(graph.get_tensor_by_name("MobilenetV1/Logits/SpatialSqueeze:0"))
    print("number class",distrib.get_shape().as_list())
    ground_labels = []
    preds_labels = []
    ids = []
    test_ids = test.id.tolist()
    for ii in range(0,len(test_ids), batch_size):
        batch = test_ids[ii:ii+batch_size]
        images,labels = hp.load_images_labels2(batch, test, path)
        preds = sess.run(distrib,{inputs:images, kp:1.0})

        ids.append(batch)
        ground_labels.append(labels)
        preds_labels.append(preds)

        if step%10 == 0:
            print(step, step*batch_size, preds[0])
        step += 1
    print(step*batch_size)
    rst_dir = "../../outputs/result/MobilenetV1_"
    hp.np.save(rst_dir+cost_type+data_bias+test_type+"D_preds", hp.np.concatenate(preds_labels))
    hp.np.save(rst_dir+cost_type+data_bias+test_type+"D_trues", hp.np.concatenate(ground_labels))
    hp.np.save(rst_dir+cost_type+data_bias+test_type+"D_ids", hp.np.concatenate(ids))
