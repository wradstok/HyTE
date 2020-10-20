from functools import partial
from models import *
from random import *
from collections import defaultdict as ddict

import numpy as np
import uuid, os, argparse
import random
import tensorflow as tf
import helper as Helper
import prediction as Pred
import sample as Sampler


YEARMIN = -50
YEARMAX = 3000
class HyTE(Model):

    def getBatches(self, data, shuffle = True):
        if shuffle: 
            random.shuffle(data)

        num_batches = len(data) // self.p.batch_size

        for i in range(num_batches):
            start_idx = i * self.p.batch_size
            yield data[start_idx : start_idx + self.p.batch_size]

    def load_data(self):
        triple_set = set()
        with open(self.p.triple2id,'r') as filein:
            for line in filein:
                h, r, t, _, _ = Helper.parse_quintuple(line)
                triple_set.add((h,r,t))

        self.start_time , self.end_time = ddict(dict), ddict(dict)
        self.inp_idx, self.start_idx, self.end_idx = ddict(list), ddict(list), ddict(list)

        # Load training data.
        train_triples, triple_time = [], ddict()
        with open(self.p.dataset,'r') as train_data:
            for i, line in enumerate(train_data):
                h, r, t, b, e = Helper.parse_quintuple(line)
                if e != -1:
                    train_triples.append([h,r,t])
                    triple_time[i] = [b, e]

        self.max_ent = Helper.get_line_count(self.p.entity2id)
        self.max_rel = Helper.get_line_count(self.p.relation2id)

        # Create year classes, e.g. (0-1853)
        self.year_list, self.year2id = Helper.create_year2id(triple_time)
        self.num_class = len(self.year2id.keys())

        self.inp_idx, self.start_idx, self.end_idx = Helper.create_id_labels(triple_time, self.year2id)

        head, rel, tail = Sampler.timestamp(self, train_triples)

        if self.p.sampler == "tdns":
            Sampler.tdns(self, head, rel, tail, triple_set)
        elif self.p.sampler == "tans":
            Sampler.tans(self, head, rel, tail, triple_set)
        else:
            raise ValueError

    def add_placeholders(self):
        self.start_year = tf.placeholder(tf.int32, shape=[None], name = 'start_time')
        self.end_year   = tf.placeholder(tf.int32, shape=[None],name = 'end_time')
        self.pos_head 	= tf.placeholder(tf.int32, [None,1])
        self.pos_tail 	= tf.placeholder(tf.int32, [None,1])
        self.rel      	= tf.placeholder(tf.int32, [None,1])
        self.neg_head 	= tf.placeholder(tf.int32, [None,1])
        self.neg_tail 	= tf.placeholder(tf.int32, [None,1])
        self.mode 	  	= tf.placeholder(tf.int32, shape = ()) # Test or train
        self.query_mode = tf.placeholder(tf.int32, shape = ()) # Entity or Relation prediction
        self.pred_mode 	= tf.placeholder(tf.int32, shape = ())  # Head or Tail

    def create_feed_dict(self, batch, dtype='train'):
        ph, pt, r, nh, nt, start_idx = zip(*batch)
        feed_dict = {}
        feed_dict[self.pos_head] = np.array(ph).reshape(-1,1)
        feed_dict[self.pos_tail] = np.array(pt).reshape(-1,1)
        feed_dict[self.rel] = np.array(r).reshape(-1,1)
        feed_dict[self.start_year] = np.array(start_idx)

        if dtype == 'train':
            feed_dict[self.neg_head] = np.array(nh).reshape(-1,1)
            feed_dict[self.neg_tail] = np.array(nt).reshape(-1,1)
            feed_dict[self.mode]   	 = 1
            feed_dict[self.pred_mode] = 0
            feed_dict[self.query_mode] = 0
        else: 
            feed_dict[self.mode] = -1

        return feed_dict

    def time_projection(self,data,t):
        inner_prod  = tf.tile(tf.expand_dims(tf.reduce_sum(data*t,axis=1),axis=1),[1,self.p.inp_dim])
        prod 		= t*inner_prod
        data = data - prod
        return data

    def add_model(self):
        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name = "ent_embedding",  shape = [self.max_ent, self.p.inp_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = False), regularizer=self.regularizer)
            self.rel_embeddings = tf.get_variable(name = "rel_embedding",  shape = [self.max_rel, self.p.inp_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = False), regularizer=self.regularizer)
            self.time_embeddings = tf.get_variable(name = "time_embedding",shape = [self.max_time, self.p.inp_dim], initializer = tf.contrib.layers.xavier_initializer(uniform =False))

        ## Some transE style model ####
        if self.p.mode == "link":
            f_test = partial(Pred.test_link_prediction, self)
        elif self.p.mode == "temporal":
            f_test = partial(Pred.test_temp_prediction, self)
        else:
            raise ValueError
        
        # mode = 1 for train mode = -1 test       
        pos_h_e, pos_t_e, pos_r_e = tf.cond(self.mode > tf.constant(0), partial(Pred.common_train, self), f_test)
        neg_h_e = tf.squeeze(tf.nn.embedding_lookup(self.ent_embeddings, self.neg_head))
        neg_t_e = tf.squeeze(tf.nn.embedding_lookup(self.ent_embeddings, self.neg_tail))

        #### ----- time -----###
        
        if self.p.onlytransE:
            # TransE model
            if self.p.L1_flag:
                pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims = True) 
                neg = tf.reduce_sum(abs(neg_h_e + pos_r_e - neg_t_e), 1, keep_dims = True) 
            else:
                pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims = True) 
                neg = tf.reduce_sum((neg_h_e + pos_r_e - neg_t_e) ** 2, 1, keep_dims = True) 
        else:
            # Hyte model
            t_1 = tf.squeeze(tf.nn.embedding_lookup(self.time_embeddings, self.start_year))

            pos_h_e_t_1 = self.time_projection(pos_h_e, t_1)
            neg_h_e_t_1 = self.time_projection(neg_h_e, t_1)
            pos_t_e_t_1 = self.time_projection(pos_t_e, t_1)
            neg_t_e_t_1 = self.time_projection(neg_t_e, t_1)
            pos_r_e_t_1 = self.time_projection(pos_r_e, t_1)

            if self.p.L1_flag:
                pos = tf.reduce_sum(abs(pos_h_e_t_1 + pos_r_e_t_1 - pos_t_e_t_1), 1, keep_dims = True) 
                neg = tf.reduce_sum(abs(neg_h_e_t_1 + pos_r_e_t_1 - neg_t_e_t_1), 1, keep_dims = True) 
            else:
                pos = tf.reduce_sum((pos_h_e_t_1 + pos_r_e_t_1 - pos_t_e_t_1) ** 2, 1, keep_dims = True) 
                neg = tf.reduce_sum((neg_h_e_t_1 + pos_r_e_t_1 - neg_t_e_t_1) ** 2, 1, keep_dims = True) 
        
        return pos, neg

    def add_loss(self, pos, neg):
        with tf.name_scope('Loss_op'):
            loss = tf.reduce_sum(tf.maximum(pos - neg + self.p.margin, 0))
            if self.regularizer != None: 
                loss += tf.contrib.layers.apply_regularization(self.regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            return loss

    def add_optimizer(self, loss):
        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(self.p.lr)
            train_op  = optimizer.minimize(loss)
        # Is this a no-op, or does assign modify state somehow?
        time_normalizer = tf.assign(self.time_embeddings, tf.nn.l2_normalize(self.time_embeddings,dim = 1))
        return train_op

    def __init__(self, params):
        self.p  = params
        self.p.batch_size = self.p.batch_size
        if self.p.l2 == 0.0: 	
            self.regularizer = None
        else:
            self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.p.l2)
        self.load_data()
        self.nbatches = len(self.data) // self.p.batch_size
        self.add_placeholders()
        self.pos, neg= self.add_model()
        self.loss      	= self.add_loss(self.pos, neg)
        self.train_op  	= self.add_optimizer(self.loss)
        self.merged_summ = tf.summary.merge_all()
        self.summ_writer = None
        print('model done')

    def run_epoch(self, sess ,data):
        losses = []

        for _, batch in enumerate(self.getBatches(data, shuffle)):
            feed = self.create_feed_dict(batch)
            l, _ = sess.run([self.loss, self.train_op], feed_dict = feed)
            losses.append(l)
        
        return np.mean(losses)

    def fit(self, sess):
        saver = tf.train.Saver(max_to_keep=None)
        check_dir = Helper.get_checkpoint_dir(self.p.name)
        
        if self.p.restore:
            save_path = os.path.join(check_dir, f'epoch_{self.p.restore_epoch}')
            saver.restore(sess, save_path)

        if not self.p.onlyTest:
            print('Start fitting')
            validation_data = Helper.read_data(self.p.valid_data)
            
            for epoch in range(self.p.max_epochs):
                loss = self.run_epoch(sess, self.data)

                if epoch % 50 == 0:
                    print(f'Epoch {epoch}\t Loss {loss}\t model {self.p.name}')

                if epoch % self.p.test_freq == 0 and epoch != 0:
                    ## -- check pointing -- ##
                    save_path = os.path.join(check_dir, f'epoch_{epoch}')   
                    saver.save(sess=sess, save_path=save_path)

                    print("Validation started")
                    if self.p.mode == "temporal":
                        Pred.temp_test_against(self, sess, validation_data, "valid", epoch)
                    elif self.p.mode == "entity":
                        Pred.test_link_against(self, sess, validation_data, "valid", epoch)
                    else:
                        raise ValueError

                    print("Validation ended")
        else: 
            print("Testing started")
            test_data = Helper.read_data(self.p.test_data)
            if self.p.mode == "temporal":
                Pred.temp_test_against(self, sess, test_data, "test", self.p.restore_epoch)
            elif self.p.mode == "entity":
                Pred.test_link_against(self, sess, test_data, "test", self.p.restore_epoch)
            else:
                raise ValueError
            print("Testing ended")

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='KG temporal inference using GCN')

    parser.add_argument('-data_type', dest= "data_type", default ='yago', choices = ['yago','wikidata', 'icews'], help ='dataset to choose')
    parser.add_argument('-version',dest = 'version', default = 'large', help = 'data version to choose')
    parser.add_argument('-test_freq', 	 dest="test_freq", 	default = 25,   	type=int, 	help='Batch size')
    parser.add_argument('-neg_sample', 	 dest="M", 		default = 1,   	type=int, 	help='Batch size')
    parser.add_argument('-gpu', 	 dest="gpu", 		default='1',			help='GPU to use')
    parser.add_argument('-name', 	 dest="name", 		default='test_'+str(uuid.uuid4()),help='Name of the run')
    parser.add_argument('-embed', 	 dest="embed_init", 	default='wiki_300',	 	help='Embedding for initialization')
    parser.add_argument('-lr',	 dest="lr", 		default=0.0001,  type=float,	help='Learning rate')
    parser.add_argument('-margin', 	 dest="margin", 	default= 10 ,   	type=float, 	help='margin')
    parser.add_argument('-batch', 	 dest="batch_size", 	default= 50000,   	type=int, 	help='Batch size')
    parser.add_argument('-epoch', 	 dest="max_epochs", 	default= 5000,   	type=int, 	help='Max epochs')
    parser.add_argument('-l2', 	 dest="l2", 		default=0.0, 	type=float, 	help='L2 regularization')
    parser.add_argument('-seed', 	 dest="seed", 		default=1234, 	type=int, 	help='Seed for randomization')
    parser.add_argument('-inp_dim',  dest="inp_dim", 	default = 128,   	type=int, 	help='Number of hidden dimensions')
    parser.add_argument('-L1_flag',  dest="L1_flag", 	action='store_false',   	 	help='Apply L1 distance norm instead of L2')
    parser.add_argument('-onlyTest', dest="onlyTest", 	action='store_true', 		help='Evaluate model on test')
    parser.add_argument('-onlytransE', dest="onlytransE", 	action='store_true', 		help='Evaluate model on only transE loss')
    parser.add_argument('-restore',	 		 dest="restore", 	    action='store_true', 		help='Restore from the previous best saved model')
    parser.add_argument('-res_epoch',	     dest="restore_epoch", 	default=200,   type =int,		help='Restore from the previous best saved model')
    parser.add_argument("-sampler", dest="sampler", choices=["tans", "tdns"], required=True)
    parser.add_argument("-pred_mode", dest="mode", choices=["entity","temporal"], default="entity")
    
    args = parser.parse_args()
    args.dataset = 'data/'+ args.data_type +'_'+ args.version+'/train.txt'
    args.entity2id = 'data/'+ args.data_type +'_'+ args.version+'/entity2id.txt'
    args.relation2id = 'data/'+ args.data_type +'_'+ args.version+'/relation2id.txt'
    args.test_data  =  'data/'+ args.data_type +'_'+ args.version+'/test.txt'
    args.valid_data  =  'data/'+ args.data_type +'_'+ args.version+'/valid.txt'
    args.triple2id  =   'data/'+ args.data_type +'_'+ args.version+'/triple2id.txt'
    args.embed_dim = int(args.embed_init.split('_')[1])

    tf.set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    Helper.set_gpu(args.gpu)
    model  = HyTE(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        print('enter fitting')
        model.fit(sess)
