import sys
sys.path.insert(0,"../..")
from nets import mobilenet_v1, vgg
import tensorflow as tf
import os
slim = tf.contrib.slim

class model():
    def __init__(self, numb_logits=10, cost="kl", model="MobilenetV1", training=True):
        self.input = tf.placeholder(tf.float32,(None,224,224,3), name="input")
        self.kp = tf.placeholder(tf.float32, name="keep_probability")
        self.lr = tf.placeholder(tf.float32, name="learning_rate")
        
        self.numb_logits = numb_logits
            
        self.target = tf.placeholder(tf.float32, (None, self.numb_logits), name="target")
        
        self.build(cost, model, training)
            
    def init_fn(self,checkpoints_dir, net_name="MobilenetV1"):
        if net_name=="MobilenetV1":
            checkpoint_exclude_scopes=[net_name+"/Logits"]
        elif net_name=="vgg_16":
            checkpoint_exclude_scopes=[net_name+"/fc8"]
    
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)

        if net_name == "MobilenetV1":
            ckpt = 'mobilenet_v1_1.0_224.ckpt'
        if net_name == "vgg_16":
            ckpt = 'vgg_16.ckpt'
        return slim.assign_from_checkpoint_fn(
          os.path.join(checkpoints_dir, ckpt),
          variables_to_restore)
    
    def build(self, cost, model, train):
        if model=="MobilenetV1":
            with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
                self.logits, self.end_points = mobilenet_v1.mobilenet_v1(self.input, num_classes=self.numb_logits,
                                                       dropout_keep_prob=self.kp, is_training=train)
        elif model=="vgg_16":
            with slim.arg_scope(vgg.vgg_arg_scope()):
                self.logits, self.end_points = vgg.vgg_16(self.input, num_classes=self.numb_logits,
                                                       dropout_keep_prob=self.kp, is_training=True)
        
        self.prob = tf.nn.softmax(self.logits, name="prob")
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.pow(self.prob - self.target,2),axis=1))
        tf.summary.scalar('loss', self.loss)
        if cost=="mse":
            self.cost = self.loss
        else:
            self.xtarget = self.target*(1-1e-11) + 1e-12
            assert self.xtarget.get_shape().as_list()[1] == self.numb_logits
            self.xprob = self.prob*(1-1e-11) + 1e-12
            assert self.xprob.get_shape().as_list()[1] == self.numb_logits
            self.cost = tf.reduce_mean(tf.reduce_sum(self.xtarget*tf.log(self.xtarget/self.prob), axis=1))
            tf.summary.scalar('cost_kl', self.cost)
                
