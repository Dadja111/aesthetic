import sys, random
sys.path.insert(0,"../..")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] =  sys.argv[1]

print('CUDA_VISIBLE_DEVICES',os.environ["CUDA_VISIBLE_DEVICES"])

from nets import mobilenet_v1, vgg
import tensorflow as tf
import helper as hp
slim = tf.contrib.slim
from datasets import dataset_utils

#get placeholder tensor for all the model
def get_inputs(n_output):
    """Declare placeholder 
    Args:
        n_output: output size of the model
    return: 
        images: placeholder for inputs images
        kp: placeholder for keeping probability value of dropout layer
        lr: placeholder for learning rate of the optimizer
        aestheticlevels: placeholder for image aesthetic rating distribution
    """
    images = tf.placeholder(tf.float32,(None,224,224,3), name='input_images')
    kp = tf.placeholder(tf.float32, name='keep_probability')
    aestheticlevels = tf.placeholder(tf.float32, (None, n_output), name='aesthetic_level')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    return images, kp, aestheticlevels, lr 

# define leaky rely:
def leakyRely(x,alpha=0.5):
    return tf.maximum(alpha * x, x)
#enhance one attribute of the image to make it aesthetically pleasing
def adapt_filter(attribute, filters, name="", alpha=0.5, train=True):
    """enhance one attribute of the image to make it aesthetically pleasing
    Args:
        attribute: image attribute to enhance
        filters: filter bank to apply on image attribute to make it aesthetically pleasing
        name: optional argument for variable scope
        alpha: parameter for leaky relu activation
        train: whether the operation is applied during training or inférence mode
    return: 
        returned enhanced version of input attribute
    """
    attrib_1 = tf.transpose(attribute, [3,1,2,0])
    filters_11 = tf.transpose(filters, [3,1,2,0,4])
    enh_1_1 = tf.nn.depthwise_conv2d(attrib_1, filters_11[0], strides=[1,1,1,1], padding="SAME", name=name+"deph1")
    enh_1_2 = tf.maximum(alpha * enh_1_1, enh_1_1)
    enh_1_3 = tf.transpose(enh_1_2, [3,1,2,0])
    bacth1 = tf.layers.batch_normalization(enh_1_3, training=train)
    enh_1_4 = tf.transpose(bacth1, [3,1,2,0])
    
    enh_2_1 = tf.nn.depthwise_conv2d(enh_1_4, filters_11[1], strides=[1,1,1,1], padding="SAME", name=name+"deph2")
    enh_2_2 = tf.maximum(alpha * enh_2_1, enh_2_1)
    enh_2_3 = tf.transpose(enh_2_2, [3,1,2,0])
    bacth2 = tf.layers.batch_normalization(enh_2_3, training=train)
    enh_2_4 = tf.transpose(bacth2, [3,1,2,0])
    
    enh_3_1 = tf.nn.depthwise_conv2d(enh_2_4, filters_11[2], strides=[1,1,1,1], padding="SAME", name=name+"deph3")
    enh_3_2 = tf.maximum(alpha * enh_3_1, enh_3_1)
    enh_3_3 = tf.transpose(enh_3_2, [3,1,2,0])
    bacth3 = tf.layers.batch_normalization(enh_3_3, training=train)
    enh_2_4 = tf.transpose(bacth3, [3,1,2,0])
    
    enh_4_1 = tf.nn.depthwise_conv2d(enh_2_4, filters_11[3], strides=[1,1,1,1], padding="SAME", name=name+"deph4")
    
    
    return tf.transpose(enh_4_1, [3,1,2,0])

# define generator model
def generator(images, n_filter=4,train=True, reuse=False):
    """define generator model
    Args:
        images: input images for generator
        n_filter: number of filter to learn from each image
        train: boolean value to specify if the network is in training mode or inference mode
        reuse: whether to reuse network variables or not    return: 
        output_2: enhanced version of input value 
    """
    with tf.variable_scope("generator",reuse=reuse):
        # first generator network
        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            logits, end_points = mobilenet_v1.mobilenet_v1(images, activation_fn=leakyRely, dropout_keep_prob=1, is_training=train)
        net = end_points['Conv2d_13_pointwise']

        filters_1 = slim.conv2d(net,256, [3, 3],stride=2, activation_fn=tf.nn.relu, padding='VALID',
                             normalizer_fn=None, scope='filters_1')
        filters_2_1 = slim.conv2d(filters_1,n_filter, [1, 1],stride=1, activation_fn=None, padding='SAME',
                             normalizer_fn=None, scope='filters_2_1')

        filters_2_2 = tf.expand_dims(filters_2_1,axis=4, name='filters_2_2')

        output_1 = adapt_filter(images[:,:,:,1:2], filters_2_2[:,:,:,0:4,:], name="sat_adapt", train=train)
        print("output_1", output_1.get_shape().as_list())
        output_2 = tf.concat([images[:,:,:,0:1], output_1,images[:,:,:,2:3] ], axis=3)
        print("output_2", output_2.get_shape().as_list())
    return output_2

# get sementics params of images
def perceptual_params(images, reuse=False):
    """get semntics params of images
    Args:
        images: input images for generator
        reuse: whether to reuse network variables or not
    return: 
        sementics params of images
    """
    with tf.variable_scope("semantic",reuse=reuse):
        # first generator network
        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            logits, end_points = mobilenet_v1.mobilenet_v1(images, num_classes=1001, dropout_keep_prob=1, is_training=False)
    return tf.squeeze(end_points['AvgPool_1a'],[1,2])


# define discriminator model
def discriminator(images, kp, n_output=10, reuse=False, train=True):
    """define discriminator model 
    Args:
        images: input images for discriminator
        kp: keeping probality for droping layer of discriminator
        n_output: discriminator output size
        reuse: whether reuse variable or not 
        train: training mode or inference mode
    return: 
        preds: discriminator output containing image aesthetic rating and other variables
    """
    with tf.variable_scope("discriminator", reuse=reuse):
        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
            logits, end_points = mobilenet_v1.mobilenet_v1(images, num_classes=n_output,
                                                       dropout_keep_prob=kp, is_training=train)
    preds = tf.nn.softmax(logits)
    print("preds: ",preds.get_shape().as_list())
    return preds

# define the symetric kullback divergence between two distribution
def kl_div(distrib_a, distrib_b, name=""):
    """define the symetric kullback divergence between two distribution
    Args:
        distrib_a: first distribution
        distrib_b: second distribution
        kp: keeping probality for droping layer of discriminator
        name: opération name
    return: 
        div: symetric kullback divergence between first and second distribution
    """
    distrib_an = distrib_a*(1-1e-11) + 1e-12
    distrib_bn = distrib_b*(1-1e-11) + 1e-12
    print("distrib_bn ",distrib_b.get_shape().as_list())
    print("distrib_an ",distrib_a.get_shape().as_list())
    assert distrib_bn.get_shape().as_list() == distrib_an.get_shape().as_list() and distrib_bn.get_shape().as_list()[1] == 10
    diva = tf.reduce_sum(distrib_an*tf.log(distrib_an/distrib_bn), axis=1)
    divb = tf.reduce_sum(distrib_bn*tf.log(distrib_bn/distrib_an), axis=1)
    print("lossa: ",diva.get_shape().as_list())
    print("lossb: ",divb.get_shape().as_list())
    div = tf.reduce_mean(diva+divb)
    print("loss: ",div.get_shape().as_list())
    tf.summary.scalar(name+"kl_loss", div)
    return div

# define earth mover distance between two distribution
def EM(distrib_a, distrib_b):
    """define earth mover distance between two distribution
    Args:
        distrib_a: first distribution
        distrib_b: second distribution
        kp: keeping probality for droping layer of discriminator
    return: 
        em_dist: earth mover distance between first and second distribution
    """
    distrib_ac = tf.cumsum(distrib_a, axis=1)
    distrib_bc = tf.cumsum(distrib_b, axis=1)
    em_dist = tf.reduce_mean(tf.reduce_sum(distrib_ac-distrib_bc,axis=1))
    print("em: ",em_dist.get_shape().as_list())
    return em_dist

# get difference between image aesthetic mean rating distribution and 10
def get_diff_mean(distrib_a, distrib_b):
    """get difference between image aesthetic mean rating distribution of two images
    Args:
        distrib_a: fisrt image aesthetic rating distribution
        distrib_b: second image aesthetic rating distribution
    return: 
        diff: difference between image aesthetic mean rating distribution and 10
    """
    w = tf.constant([1,2,3,4,5,6,7,8,9,10], dtype=tf.float32)
    mean = tf.reduce_sum(w*(distrib_a - distrib_b),axis=1)
    print("mean...")
    return tf.reduce_mean(mean)

# get variance of image aesthetic rating distribution
def get_var(distrib):
    """get variance of image aesthetic rating distribution
    Args:
        distrib: image aesthetic rating distribution
    return: 
        var: variance of image aesthetic rating distribution
    """
    w = tf.constant([1,2,3,4,5,6,7,8,9,10], dtype=tf.float32)
    var = tf.reduce_sum((w**2)*distrib, axis=1) - tf.reduce_sum(w*distrib,axis=1)**2
    return var

# define generator and discriminator loss
def model_loss(x, kp, target):
    """define generator and discriminator loss
    Args:
        x: images for generator and discriminator input
        target: image aesthetic rating distribution
    return: 
        d_loss: discriminator loss
        g_loss: generator loss
    """
    g_model = generator(x)
    d_model_real = discriminator(x, kp)
    d_model_fake = discriminator(g_model,kp, reuse=True)

    percept_real = perceptual_params(x)
    percept_fake = perceptual_params(g_model, reuse=True)
   
    d_loss_pred = kl_div(d_model_real[:,0:10], target, name="den_real_")
    
    d_loss = d_loss_pred

    g_percept = tf.reduce_mean(tf.reduce_mean((percept_real-percept_fake)**2, axis=1))
    
    g_loss_enh = EM(d_model_fake, target)
    g_loss_var = tf.reduce_mean(get_var(d_model_fake))
    
    g_loss = g_loss_enh + g_loss_var + g_percept
    #g_loss = get_diff_mean(target, d_model_fake)
    
    return d_loss, g_loss

# define generator and discriminator optimization
def model_opt(d_loss, g_loss, lr):
    """define generator and discriminator optimization
    Args:
        d_loss: discriminator loss
        g_loss: generator loss
    return: 
        d_train_opt: discriminator optimization
        g_train_opt: generator optimization
    """
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith("discriminator")]
    g_vars = [var for var in t_vars if var.name.startswith("generator")]
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(lr).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(lr).minimize(g_loss, var_list=g_vars)
    return d_train_opt, g_train_opt

# define class for aesthetic enhancing GAN model
class GANM:
    def __init__(self, n_output):
        """define class for aesthetic enhancing GAN model
        Args:
           n_output: discriminator output size
        """
        tf.reset_default_graph()
        self.input_rgb, self.kp, self.target, self.lr = get_inputs(n_output)
        self.input = (tf.image.rgb_to_hsv((self.input_rgb+1)*0.5)-0.5)*2
        self.d_loss, self.g_loss = model_loss(self.input, self.kp, self.target)
        self.d_opt, self.g_opt = model_opt(self.d_loss, self.g_loss, self.lr)

# initialize model parameter with paremeters obtained after training on imagenet
def init_fn(checkpoints_dir, net_name="", ckpt = 'mobilenet_v1_1.0_224.ckpt', exclude=True):
    """initialize model parameter with paremeters obtained after training on imagenet
    Args:
        checkpoints_dir: checkpoints directory
        net_name: variable scope
    return: 
        list of variable to initialize
    """
    if exclude:
        checkpoint_exclude_scopes=[net_name+"MobilenetV1/Logits"]
    else:
        checkpoint_exclude_scopes=[]

    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = {}
    for var in slim.get_model_variables(net_name+"MobilenetV1"):
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore[var.op.name.replace(net_name, "")] = var
    
    return slim.assign_from_checkpoint_fn(
      os.path.join(checkpoints_dir, ckpt),
      variables_to_restore)

# declaring of global variables
dis_ckpt_dir = "../../tmp/ckpts/final/random/"
checkpoints_dir = '../../tmp/checkpoints' #director where to store the pretraining model weights
url = "http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz" #url for downloading pretraining model weights 
#dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir) # download pretraining model weights and save them

path = "../../../../../dataset/AVA/" #specify the path to folder containing images folder. For example here AVA folder contains folder name images which contains images.
save_dir = "../../tmp/sat4/" # where to save the output of training model

data = hp.pd.read_csv("../../../../data/train.csv") # load images meta data file
train = data[0:10000] # sample image for training
test = data[30000:30400] # sample image for testing
kp = 0.99 # dropout layer keep probability. it is used here for discriminator network
learning_rate = 1e-7 # learning rate for discriminator
learning_rate2 = 1e-6 # learning rate for generator
print_step = 50 # number of training step after which to print discriminator and generator loss
batch_size = 32 # batch size
epochs = 10 # epoch
gan = GANM(10) # create traing graph
random.seed(2017) 
hp.np.random.seed(2017)
tf.set_random_seed(2017)
gen_init_fn = init_fn(checkpoints_dir, net_name="generator/") # initialize generator
per_init_fn = init_fn(checkpoints_dir, net_name="semantic/", exclude=False) # initialize perceptual network
dis_init_fn = init_fn(dis_ckpt_dir, net_name="discriminator/",ckpt="aesthetic_kl.ckpt", exclude=False) # initialize discriminator
saver = tf.train.Saver()
j = 0
train_ids = train.id.tolist()
test_ids = test.id.tolist()
step = 0

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    gen_init_fn(sess)
    per_init_fn(sess)
    dis_init_fn(sess)
    dl = []
    gl = []
    for e in range(epochs):
        random.shuffle(train_ids)
        print(e, "learning rate: ",learning_rate, learning_rate2)
        for ii in range(0,len(train_ids), batch_size):
            batch = train_ids[ii:ii+batch_size]
            images,labels = hp.load_images_labels2(batch, train, path)
            _ = sess.run(gan.d_opt,{gan.input:images, gan.target:labels, gan.kp:kp, gan.lr:learning_rate})
            _ = sess.run(gan.g_opt,{gan.input:images, gan.target:labels, gan.kp:kp, gan.lr:learning_rate2})

            if step%print_step == 0:
                jn = 0
                g_l = 0
                d_l = 0
                for j in range(0,len(test_ids), batch_size):
                    batch = test_ids[j:j+batch_size]
                    images,labels = hp.load_images_labels2(batch, test, path)
                    dis_loss, gen_loss = sess.run([gan.d_loss, gan.g_loss],{gan.input:images, gan.target:labels, gan.kp:1.0})
                    d_l += dis_loss
                    g_l += gen_loss
                    jn += 1
                d_l /=jn
                g_l/=jn
                print(e, step,"dis_loss", d_l,"gen_loss", g_l)
                dl.append(d_l)
                gl.append(g_l)
            step +=1
        batch = test_ids[0:batch_size]
        images,labels = hp.load_images_labels2(batch, test, path)
        enh_images = sess.run(generator(gan.input, reuse=True, train=False),{gan.input:images, gan.kp:1.0})
        hp.np.save(save_dir+"labels_"+str(step),labels) #save image aesthetic rating
        hp.np.save(save_dir+"images_"+str(step),images) #save images
        hp.np.save(save_dir+"enh_images_"+str(step),enh_images) #save enhanced version of image by generator network
        hp.np.save(save_dir+"dl_"+str(step),dl) # save discriminator loss after each epoch
        hp.np.save(save_dir+"gl_"+str(step),gl) # save generator loss after each epoch

        if (e+1)%3==0:
            learning_rate /=10
            learning_rate2 /=10
    batch = train_ids[0:batch_size]
    images,labels = hp.load_images_labels2(batch, train, path)
    enh_images = sess.run(generator(gan.input, reuse=True, train=False),{gan.input:images, gan.kp:1.0})
    hp.np.save(save_dir+"dl",dl) # save discriminator loss
    hp.np.save(save_dir+"gl",gl) # save generator loss
    hp.np.save(save_dir+"labels",labels) #save image aesthetic rating
    hp.np.save(save_dir+"images",images) #save images
    hp.np.save(save_dir+"enh_images",enh_images) #save enhanced version of image by generator network

    saver.save(sess,save_dir+"final_model.ckpt") #save network graph and weight






