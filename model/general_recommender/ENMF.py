'''
Reference: Tong Zhao et al., "Leveraging Social Connections to Improve 
Personalized Ranking for Collaborative Filtering." in CIKM 2014
@author: wubin
'''
import tensorflow as tf
import numpy as np
from time import time
from util import learner
from model.AbstractRecommender import AbstractRecommender
from util.data_iterator import DataIterator
from util.tool import pad_sequences, csr_to_user_dict, timer

class ENMF(AbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(ENMF, self).__init__(dataset, conf)
        self.learning_rate = conf["learning_rate"]
        self.embedding_size = conf["embedding_size"]
        self.learner = conf["learner"]
        self.num_epochs= conf["epochs"]
        self.batch_size = conf["batch_size"]
        self.verbose = conf["verbose"]
        self.alpha = conf["alpha"]
        self.reg = conf["reg"]
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.train_dict = csr_to_user_dict(self.dataset.train_matrix)
        self.sess = sess

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, [None,], name="user_input")
    
            self.items_input =tf.placeholder(tf.int32, [None, None], name="items_input")
            
    def _create_variables(self):
        self.user_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0,
                                                    stddev=0.01), dtype=tf.float32, name="user_embed")
       
        self.c1 = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0,
                                                    stddev=0.01), dtype=tf.float32, name="c1")
        self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
        
        self.item_embeddings = tf.concat([self.c1,self.c2], 0, name='item_embed')
        
        # item domain
        self.H_i = tf.Variable(tf.constant(0.01, shape=[self.embedding_size, 1]), name="hi")
    
        
    def _create_inference(self):
        with tf.name_scope("inference"):
            self.uid_A = tf.nn.embedding_lookup(self.user_embeddings, self.user_input)
    
            self.pos_item = tf.nn.embedding_lookup(self.item_embeddings,self.items_input)
            self.pos_num_r = tf.cast(tf.not_equal(self.items_input, self.num_items), 'float32')
            self.pos_item = tf.einsum('ab,abc->abc', self.pos_num_r, self.pos_item)
           
            
            self.pos_r=tf.einsum('ac,abc->abc',self.uid_A,self.pos_item)
            self.pos_r=tf.einsum('ajk,kl->ajl', self.pos_r, self.H_i)
            self.pos_r = tf.reshape(self.pos_r, [-1, tf.shape(self.pos_item)[1]])
            

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.loss1=self.alpha*tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.einsum('ab,ac->abc',self.item_embeddings,self.item_embeddings),0)
                                    *tf.reduce_sum(tf.einsum('ab,ac->abc',self.user_embeddings,self.user_embeddings),0)
                                    *tf.matmul(self.H_i,self.H_i,transpose_b=True),0),0)
            
            self.loss1+=tf.reduce_sum((1.0 - self.alpha) * tf.square(self.pos_r) - 2.0 * self.pos_r)
    
            self.loss=self.loss1 +self.reg*tf.nn.l2_loss(self.user_embeddings)\
                      +self.reg*tf.nn.l2_loss(self.item_embeddings)\
                      +self.reg*tf.nn.l2_loss(self.H_i)
    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate)
    
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()
    
#---------- training process -------
    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.num_epochs):
            user_train, item_train = self._get_input_data()
            data_iter = DataIterator(user_train, item_train,
                                     batch_size=self.batch_size, shuffle=True)
            total_loss = 0.0
            training_start_time = time()
            num_training_instances = len(user_train)
            for bat_user_train, bat_item_train, in data_iter:
                bat_item_train = pad_sequences(bat_item_train, value=self.num_items)
                    
                feed_dict = {self.user_input:bat_user_train,
                             self.items_input:bat_item_train}
                      
                loss,_ = self.sess.run((self.loss,self.optimizer),feed_dict=feed_dict)
                total_loss+=loss
            self.logger.info("[iter %d : loss : %f, time: %f]" % (epoch, total_loss/num_training_instances,
                                                             time()-training_start_time))
            if epoch % self.verbose == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))
    
    @timer
    def evaluate(self):
        return self.evaluator.evaluate(self)

    def _get_input_data(self):
        user_train, item_train = [], []
        for u in range(self.num_users):
            items_by_user = self.train_dict[u]
            user_train.append(u)
            item_train.append(items_by_user)
    
        user_train = np.array(user_train)
        item_train = np.array(item_train)
        
        num_training_instances = len(user_train)
        shuffle_index = np.arange(num_training_instances,dtype=np.int32)
        np.random.shuffle(shuffle_index)
        user_train = user_train[shuffle_index]
        item_train = item_train[shuffle_index]
        
        return user_train, item_train
            
    def predict(self, user_ids, candidate_items_userids):
        ratings = []
        if candidate_items_userids is not None:
            print("1233")
        else:
            candidate_items_userids = np.arange(self.num_items)
            for user_id in user_ids:
                eval_items = np.array(candidate_items_userids)
                eval_items = eval_items[np.newaxis,:]
                result = self.sess.run(self.pos_r,
                                   feed_dict={self.user_input: [user_id], 
                                   self.items_input:eval_items})   
                ratings.append(np.reshape(result, [-1]))
        return ratings