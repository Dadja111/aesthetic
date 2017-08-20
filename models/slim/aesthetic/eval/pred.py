import sys
sys.path.insert(0,"../utils")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] =  sys.argv[1]

print('CUDA_VISIBLE_DEVICES',os.environ["CUDA_VISIBLE_DEVICES"])

import helper as hp
import random
import tensorflow as tf
import model as model
slim = tf.contrib.slim

cost_type= sys.argv[2] #"mse" #possible values: mse, kl
data_bias= sys.argv[3] #"weigthed" #possible values: balancing, weigthed, NW
test_type = sys.argv[4] # kl balancing E MobilenetV1
model_type = sys.argv[5]
print('cost type :',cost_type)
print('data bias :', data_bias)
print('test type :',test_type)
print('model type :', model_type)
path_prefix = "../../../../"
path = path_prefix+"../dataset/AVA/"

if "V" == test_type:
    test = hp.pd.read_csv(path_prefix+"data/val.csv")
else:
    test = hp.pd.read_csv(path_prefix+"data/"+test_type+"D_test.csv")

batch_size = 300 if model_type == "MobilenetV1" else 128
print(batch_size)
checkpoints_dir = '../../outputs/ckpts/final/'+data_bias if model_type == "MobilenetV1" else '../../outputs/ckpts/vgg16/final/'+data_bias
ckpt = "aesthetic_"+cost_type+".ckpt" 

print("evaluation...")
tf.reset_default_graph()
with tf.Graph().as_default():
    md = model.model(10, cost=cost_type, model=model_type, training=False)
    init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, ckpt),
                                             slim.get_model_variables(model_type))
    ids = []
    ground_labels = []
    preds_labels = []

    with tf.device("/gpu:0"):
        with tf.Session() as sess:
            init_fn(sess)
            test_ids = test.id.tolist()
            step=0
            for i in range(0,len(test_ids), batch_size):
                batch = test_ids[i:i+batch_size]
                images,labels = hp.load_images_labels2(batch, test, path)
                preds = sess.run(md.prob,{md.input:images, md.kp:1.0})

                ids.append(batch)
                ground_labels.append(labels)
                preds_labels.append(preds)

                if step%10 == 0:
                    print(step, step*batch_size, preds[0])
                step += 1
    print(step*batch_size)
    rst_dir = "../../outputs/result/"
    hp.np.save(rst_dir+model_type+cost_type+data_bias+test_type+"D_preds", hp.np.concatenate(preds_labels))
    hp.np.save(rst_dir+model_type+cost_type+data_bias+test_type+"D_trues", hp.np.concatenate(ground_labels))
    hp.np.save(rst_dir+model_type+cost_type+data_bias+test_type+"D_ids", hp.np.concatenate(ids))

