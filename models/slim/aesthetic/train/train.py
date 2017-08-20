import sys
sys.path.insert(0,"../utils")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
gpu_number = 2
os.environ["CUDA_VISIBLE_DEVICES"] =  sys.argv[1]

print('CUDA_VISIBLE_DEVICES',os.environ["CUDA_VISIBLE_DEVICES"])

import helper as hp
import random
import tensorflow as tf
import model as model

cost_type= sys.argv[2] #possible cost function type: mse, kl
data_bias= sys.argv[3] #possible batch formation type: balancing, random
print('cost type :',cost_type)
print('batch formation type :', data_bias)
learning_rate = 1e-5 #1e-4 #1e-5, 1e-8
kp = 0.9 # dropout keep probability

path_prefix = "../../../../"
path = path_prefix+"../dataset/AVA/"
save_step = 2000
v_step = 50
print_step = 200
epoch_number = 10
iterations_number = 31000

if not hp.os.path.isdir(path_prefix+"data"):
    hp.os.mkdir(path_prefix+"data")
    train, val, RD, ED = hp.get_train_test(hp.load_images_metadata(path))
    train.to_csv(path_prefix+"data/train.csv", index=False)
    val.to_csv(path_prefix+"data/val.csv", index=False)
    RD.to_csv(path_prefix+"data/RD_test.csv", index=False)
    ED.to_csv(path_prefix+"data/ED_test.csv", index=False)
else:
    train = hp.pd.read_csv(path_prefix+"data/train.csv")
    val = hp.pd.read_csv(path_prefix+"data/val.csv")

if data_bias=="balancing":
    if sys.argv[5] == "MobilenetV1":
        mini_size = 12
    elif sys.argv[5] == "vgg_16":
        mini_size = 4

    clusters = hp.get_train_clusters(train, 64)
    n_clusters = len(clusters.keys())
    clusters_val = hp.get_train_clusters(val, mini_size)
    n_clusters_val = len(clusters_val.keys())

    print("Train number of cluster",n_clusters)
    print("Val number of cluster",n_clusters_val)
    
    batch_size = n_clusters*mini_size
    cluster_mini_batches = {k: hp.get_mini_bacth(clusters[k], mini_size) for k in clusters.keys()}
    cluster_mini_batches_val = {k: hp.get_mini_bacth(clusters_val[k], mini_size)
                                for k in clusters_val.keys()}
else:
    if sys.argv[5] == "MobilenetV1":
        batch_size = 128
    elif sys.argv[5] == "vgg_16":
        batch_size = 64
    
print(batch_size)

#url = "http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz"
if sys.argv[5] == "MobilenetV1":
    checkpoints_dir = '../../tmp/checkpoints'
    save_dir = "../../outputs/ckpts/"
    print("mobilenet")
elif sys.argv[5] == "vgg_16":
    checkpoints_dir = '../../tmp/vgg'
    save_dir = "../../outputs/ckpts/vgg16/"
    kp = 0.5
    print("vgg-16")
if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)

#dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

print("training...")
tf.reset_default_graph()
with tf.Graph().as_default(): 
    md = model.model(10, cost=cost_type, model=sys.argv[5])
    optimiser = tf.train.AdamOptimizer(md.lr).minimize(md.cost)
    #optimiser = tf.train.GradientDescentOptimizer(md.lr).minimize(md.cost)
    merged = tf.summary.merge_all()
    init_fn = md.init_fn(checkpoints_dir, net_name=sys.argv[5])
    saver = tf.train.Saver()
    train_loss_hist = []
    val_loss_hist = []
    lr_hist = []
    with tf.device("/gpu:0"):    
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter('../../outputs/summary/train_2'+cost_type,sess.graph)
            val_writer = tf.summary.FileWriter('../../outputs/summary/test_2'+cost_type)
            sess.run(tf.global_variables_initializer())
            init_fn(sess)
		
            if data_bias=="balancing":
                iterations = iterations_number
                for step in range(iterations):
                    batch = hp.get_batch(cluster_mini_batches, mini_size, clusters)
                    images,labels = hp.load_images_labels2(batch, train, path)
                    sess.run(optimiser,{md.input:images, md.target:labels, md.kp:kp, md.lr:learning_rate})

                    if step%print_step == 0:
                        train_loss, train_summary = sess.run([md.loss, merged],{md.input:images, md.target:labels, md.kp:1.0})
                        train_writer.add_summary(train_summary, step)
                        train_loss_hist.append([step,train_loss])
                        batch = hp.get_batch(cluster_mini_batches_val, mini_size, clusters_val)
                        images,labels = hp.load_images_labels2(batch, val, path)
                        val_loss, val_summary = sess.run([md.loss, merged],{md.input:images, md.target:labels, md.kp:1.0})
                        val_writer.add_summary(val_summary, step)
                        val_loss_hist.append([step,val_loss])
                        print("iteration ",step,"/",iterations,"train error ",train_loss, "\tvalidation error",val_loss)
                        if len(train_loss_hist) >= 2 and train_loss_hist[len(train_loss_hist)-1][1] > train_loss_hist[len(train_loss_hist)-2][1]:
                            if learning_rate > 1e-12:
                                learning_rate /10

                    elif step%v_step == 0:
                        train_loss = sess.run(md.loss,{md.input:images, md.target:labels, md.kp:1.0})
                        batch = hp.get_batch(cluster_mini_batches_val, mini_size, clusters_val)
                        images,labels = hp.load_images_labels2(batch, val, path)
                        val_loss = sess.run(md.loss,{md.input:images, md.target:labels, md.kp:1.0})
                        print("iteration ",step,"/",iterations,"train error ",train_loss, "\tvalidation error",val_loss)

                    if step%save_step == 0  and sys.argv[4] != "test":
                        saver.save(sess,save_dir+"iterations/"+data_bias+"/aesthetic_"+cost_type+str(step)+".ckpt")
            else:
                epochs = epoch_number
                j = 0
                train_ids = train.id.tolist()
                val_ids = val.id.tolist()
                step = 0
		        
                for e in range(epochs):
                    random.shuffle(train_ids)
                    for ii in range(0,len(train_ids), batch_size):
                         batch = train_ids[ii:ii+batch_size]
                         images,labels = hp.load_images_labels2(batch, train, path)
                         sess.run(optimiser,{md.input:images, md.target:labels, md.kp:kp, md.lr:learning_rate})

                         if step%print_step == 0:
                             train_loss, train_summary = sess.run([md.loss, merged],{md.input:images, md.target:labels, md.kp:1.0})
                             train_writer.add_summary(train_summary, step)
                             train_loss_hist.append([step,train_loss])
                             batch = val_ids[j:j+batch_size]
                             j += batch_size
                             if j >= len(val_ids):
                                 j=0
                             images,labels = hp.load_images_labels2(batch, val, path)
                             val_loss, val_summary = sess.run([md.loss, merged],{md.input:images, md.target:labels, md.kp:1.0})
                             val_writer.add_summary(val_summary, step)
                             val_loss_hist.append([step,val_loss])
                             print("iteration ",e,step,"/",epochs,"train error ",train_loss, "\tvalidation error",val_loss)
 
                             if len(train_loss_hist) >= 2 and train_loss_hist[len(train_loss_hist)-1][1] > train_loss_hist[len(train_loss_hist)-2][1]:
                                 if learning_rate > 1e-12:
                                     learning_rate /=10
                         elif step%v_step == 0:
                             #print("is_dir",hp.os.path.isdir(".ckpts/iterations/"+data_bias),".ckpts/iterations/"+data_bias)
                             train_loss = sess.run(md.loss,{md.input:images, md.target:labels, md.kp:1.0})
                             batch = val_ids[j:j+batch_size]
                             j += batch_size
                             if j >= len(val_ids):
                                 j=0
                             images,labels = hp.load_images_labels2(batch, val, path)
                             val_loss = sess.run(md.loss,{md.input:images, md.target:labels, md.kp:1.0})
                             print("iteration ",e,step,"/",epochs,"train error ",train_loss, "\tvalidation error",val_loss)

                         if step%save_step == 0  and sys.argv[4] != "test":
                             saver.save(sess,save_dir+"iterations/"+data_bias+"/aesthetic_"+cost_type+str(e)+str(step)+".ckpt")
                         step +=1

            hp.np.save(save_dir+"final/"+data_bias+"/train_loss_hist_"+cost_type, train_loss_hist)
            hp.np.save(save_dir+"final/"+data_bias+"/val_loss_hist_"+cost_type, val_loss_hist)

            saver.save(sess,save_dir+"final/"+data_bias+"/aesthetic_"+cost_type+".ckpt")
             
