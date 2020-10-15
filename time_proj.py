from models import *
from random import *
from collections import defaultdict as ddict

import uuid, os, argparse
import random
import tensorflow as tf
import numpy as np

import helper as Helper
import prediction as Pred


YEARMIN = -50
YEARMAX = 3000
class HyTE(Model):
	def getBatches(self, data, shuffle = True):
		if shuffle: random.shuffle(data)
		num_batches = len(data) // self.p.batch_size

		for i in range(num_batches):
			start_idx = i * self.p.batch_size
			yield data[start_idx : start_idx + self.p.batch_size]

	def load_data(self):
		triple_set = []
		with open(self.p.triple2id,'r') as filein:
			for line in filein:
				tup = (int(line.split()[0].strip()) , int(line.split()[1].strip()), int(line.split()[2].strip()))
				triple_set.append(tup)
		triple_set=set(triple_set)

		train_triples = []
		self.start_time , self.end_time, self.num_class  = ddict(dict), ddict(dict), ddict(dict)
		triple_time = dict()
		self.inp_idx, self.start_idx, self.end_idx ,self.labels = ddict(list), ddict(list), ddict(list), ddict(list)
		
		# Load training data.
		with open(self.p.dataset,'r') as train_data:
			for i, line in enumerate(train_data):
				h, r, t, b, e = Helper.parse_quintuple(line)
				if e != -1:
					train_triples.append([h,r,t])
					triple_time[i] = [b, e]

		self.max_ent = Helper.get_line_count(self.p.entity2id)
		self.max_rel = Helper.get_line_count(self.p.relation2id)

		self.year_list, self.year2id = Helper.create_year2id(triple_time)
		self.num_class = len(self.year2id.keys())

		self.inp_idx, self.start_idx, self.end_idx = Helper.create_id_labels(triple_time, self.year2id)
		self.num_class = len(self.year2id.keys())

		keep_idx = set(self.inp_idx)
		for i in range (len(train_triples)-1,-1,-1):
			if i not in keep_idx:
				del train_triples[i]
		
		posh, rela, post =  map(list, zip(*train_triples))
		head, rel, tail = map(list, zip(*train_triples))

		for i in range(len(posh)):
			if self.start_idx[i] < self.end_idx[i]:
				for j in range(self.start_idx[i] + 1,self.end_idx[i] + 1):
					head.append(posh[i])
					rel.append(rela[i])
					tail.append(post[i])
					self.start_idx.append(j)

		self.ph, self.pt, self.r,self.nh, self.nt , self.triple_time  = [], [], [], [], [], []
		for triple in range(len(head)):
			neg_set = set()
			for k in range(self.p.M):
				possible_head = randint(0,self.max_ent-1)
				while (possible_head, rel[triple], tail[triple]) in triple_set or (possible_head, rel[triple],tail[triple]) in neg_set:
					possible_head = randint(0,self.max_ent-1)
				self.nh.append(possible_head)
				self.nt.append(tail[triple])
				self.r.append(rel[triple])
				self.ph.append(head[triple])
				self.pt.append(tail[triple])
				self.triple_time.append(self.start_idx[triple])
				neg_set.add((possible_head, rel[triple],tail[triple]))
		
		for triple in range(len(tail)):
			neg_set = set()
			for k in range(self.p.M):
				possible_tail = randint(0,self.max_ent-1)
				while (head[triple], rel[triple],possible_tail) in triple_set or (head[triple], rel[triple],possible_tail) in neg_set:
					possible_tail = randint(0,self.max_ent-1)
				self.nh.append(head[triple])
				self.nt.append(possible_tail)
				self.r.append(rel[triple])
				self.ph.append(head[triple])
				self.pt.append(tail[triple])
				self.triple_time.append(self.start_idx[triple])
				neg_set.add((head[triple], rel[triple],possible_tail))

		self.max_time = len(self.year2id.keys())
		self.data = list(zip(self.ph, self.pt, self.r , self.nh, self.nt, self.triple_time))
		self.data = self.data + self.data[0:self.p.batch_size]

	def add_placeholders(self):
		self.start_year = tf.placeholder(tf.int32, shape=[None], name = 'start_time')
		self.end_year   = tf.placeholder(tf.int32, shape=[None],name = 'end_time')
		self.pos_head 	= tf.placeholder(tf.int32, [None,1])
		self.pos_tail 	= tf.placeholder(tf.int32, [None,1])
		self.rel      	= tf.placeholder(tf.int32, [None,1])
		self.neg_head 	= tf.placeholder(tf.int32, [None,1])
		self.neg_tail 	= tf.placeholder(tf.int32, [None,1])
		self.mode 	  	= tf.placeholder(tf.int32, shape = ())
		self.pred_mode 	= tf.placeholder(tf.int32, shape = ())
		self.query_mode = tf.placeholder(tf.int32, shape = ())

	def create_feed_dict(self, batch, wLabels=True,dtype='train'):
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
			#nn_in = self.input_x
		with tf.name_scope("embedding"):
			self.ent_embeddings = tf.get_variable(name = "ent_embedding",  shape = [self.max_ent, self.p.inp_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = False), regularizer=self.regularizer)
			self.rel_embeddings = tf.get_variable(name = "rel_embedding",  shape = [self.max_rel, self.p.inp_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = False), regularizer=self.regularizer)
			self.time_embeddings = tf.get_variable(name = "time_embedding",shape = [self.max_time, self.p.inp_dim], initializer = tf.contrib.layers.xavier_initializer(uniform =False))

		transE_in_dim = self.p.inp_dim
		transE_in     = self.ent_embeddings
		####################------------------------ time aware GCN MODEL ---------------------------##############


	
		## Some transE style model ####
		neutral = tf.constant(0)      ## mode = 1 for train mode = -1 test
		test_type = tf.constant(0)    ##  pred_mode = 1 for head -1 for tail
		query_type = tf.constant(0)   ## query mode  =1 for head tail , -1 for rel
		
		def f_train():
			pos_h_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_head))
			pos_t_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_tail))
			pos_r_e = tf.squeeze(tf.nn.embedding_lookup(self.rel_embeddings, self.rel))
			return pos_h_e, pos_t_e, pos_r_e
		
		def f_test():
			def head_tail_query():
				def f_head():
					e2 = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_tail))
					pos_h_e = transE_in
					pos_t_e = tf.reshape(tf.tile(e2,[self.max_ent]),(self.max_ent, transE_in_dim))
					return pos_h_e, pos_t_e
				
				def f_tail():
					e1 = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_head))
					pos_h_e = tf.reshape(tf.tile(e1,[self.max_ent]),(self.max_ent, transE_in_dim))
					pos_t_e = transE_in
					return pos_h_e, pos_t_e

				pos_h_e, pos_t_e  = tf.cond(self.pred_mode > test_type, f_head, f_tail)
				r  = tf.squeeze(tf.nn.embedding_lookup(self.rel_embeddings,self.rel))
				pos_r_e = tf.reshape(tf.tile(r,[self.max_ent]),(self.max_ent,transE_in_dim))
				return pos_h_e, pos_t_e, pos_r_e
			
			def rel_query():
				e1 = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_head))
				e2 = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_tail))
				pos_h_e = tf.reshape(tf.tile(e1,[self.max_rel]),(self.max_rel, transE_in_dim))
				pos_t_e = tf.reshape(tf.tile(e2,[self.max_rel]),(self.max_rel, transE_in_dim))
				pos_r_e = self.rel_embeddings
				return pos_h_e, pos_t_e, pos_r_e

			pos_h_e, pos_t_e, pos_r_e = tf.cond(self.query_mode > query_type, head_tail_query, rel_query)
			return pos_h_e, pos_t_e, pos_r_e

		pos_h_e, pos_t_e, pos_r_e = tf.cond(self.mode > neutral, f_train, f_test)
		neg_h_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.neg_head))
		neg_t_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.neg_tail))

		if self.p.onlytransE:
			### TransE model
			if self.p.L1_flag:
				pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims = True) 
				neg = tf.reduce_sum(abs(neg_h_e + pos_r_e - neg_t_e), 1, keep_dims = True) 
			else:
				pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims = True) 
				neg = tf.reduce_sum((neg_h_e + pos_r_e - neg_t_e) ** 2, 1, keep_dims = True) 
		else:
			### Hyte model
			t_1 = tf.squeeze(tf.nn.embedding_lookup(self.time_embeddings, self.start_year))
			pos_h_e_t_1 = self.time_projection(pos_h_e,t_1)
			neg_h_e_t_1 = self.time_projection(neg_h_e,t_1)
			pos_t_e_t_1 = self.time_projection(pos_t_e,t_1)
			neg_t_e_t_1 = self.time_projection(neg_t_e,t_1)
			pos_r_e_t_1 = self.time_projection(pos_r_e,t_1)
			if self.p.L1_flag:
				pos = tf.reduce_sum(abs(pos_h_e_t_1 + pos_r_e_t_1 - pos_t_e_t_1), 1, keep_dims = True) 
				neg = tf.reduce_sum(abs(neg_h_e_t_1 + pos_r_e_t_1 - neg_t_e_t_1), 1, keep_dims = True) 
			else:
				pos = tf.reduce_sum((pos_h_e_t_1 + pos_r_e_t_1 - pos_t_e_t_1) ** 2, 1, keep_dims = True) 
				neg = tf.reduce_sum((neg_h_e_t_1 + pos_r_e_t_1 - neg_t_e_t_1) ** 2, 1, keep_dims = True) 
		
		return pos, neg

	def add_loss(self, pos, neg):
		with tf.name_scope('Loss_op'):
			loss     = tf.reduce_sum(tf.maximum(pos - neg + self.p.margin, 0))
			if self.regularizer != None: 
				loss += tf.contrib.layers.apply_regularization(self.regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
			return loss

	def add_optimizer(self, loss):
		with tf.name_scope('Optimizer'):
			optimizer = tf.train.AdamOptimizer(self.p.lr)
			train_op  = optimizer.minimize(loss)
		time_normalizer = tf.assign(self.time_embeddings, tf.nn.l2_normalize(self.time_embeddings,dim = 1))
		return train_op

	def __init__(self, params):
		self.p  = params
		self.p.batch_size = self.p.batch_size
		if self.p.l2 == 0.0: 	self.regularizer = None
		else: 			self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.p.l2)
		self.load_data()
		self.nbatches = len(self.data) // self.p.batch_size
		self.add_placeholders()
		self.pos, neg   = self.add_model()
		self.loss      	= self.add_loss(self.pos, neg)
		self.train_op  	= self.add_optimizer(self.loss)
		self.merged_summ = tf.summary.merge_all()
		self.summ_writer = None
		print('model done')

	def run_epoch(self, sess,data):
		losses = []

		for step, batch in enumerate(self.getBatches(data, shuffle)):
			feed = self.create_feed_dict(batch)
			l, a = sess.run([self.loss, self.train_op],feed_dict = feed)
			losses.append(l)
		return np.mean(losses)

	def fit(self, sess):
		saver = tf.train.Saver(max_to_keep=None)
		save_dir = 'checkpoints/' + self.p.name + '/'
		if not os.path.exists(save_dir): 
			os.makedirs(save_dir)
		save_dir_results = './results/'+ self.p.name + '/'
		
		if not os.path.exists(save_dir_results): 
			os.makedirs(save_dir_results)
		
		if self.p.restore:
			save_path = os.path.join(save_dir, f'epoch_{self.p.restore_epoch}')
			saver.restore(sess, save_path)
		
		if not self.p.onlyTest:
			print('start fitting')
			validation_data = Helper.read_data(self.p.valid_data)

			for epoch in range(self.p.max_epochs):
				l = self.run_epoch(sess,self.data)
				if epoch%50 == 0:
					print(f'Epoch {epoch}\tLoss {l}\t model {self.p.name}')
				
				if epoch % self.p.test_freq == 0 and epoch != 0:
					## -- check pointing -- ##
					save_path = os.path.join(save_dir, f'epoch_{epoch}') 
					saver.save(sess=sess, save_path=save_path)
					
					print("Validation started")
					Pred.test_link_against(self, sess, "valid", epoch, validation_data)
					print("Validation ended")

		else:
			print('Testing started')
			test_data = Helper.read_data(self.p.test_data)
			Pred.test_link_against(self, sess, "test", self.p.restore_epoch, test_data)
			print("Test ended")

if __name__== "__main__":
	print('here in main')
	parser = argparse.ArgumentParser(description='HyTE')

	parser.add_argument('-data_type', dest= "data_type", default ='yago', choices = ['yago','wikidata', 'icews'], help ='dataset to choose')
	parser.add_argument('-version',dest = 'version', default = 'large', choices = ['inter', 'intra','both','none','src'], help = 'data version to choose')
	parser.add_argument('-test_freq', 	 dest="test_freq", 	default = 25,   	type=int, 	help='Batch size')
	parser.add_argument('-neg_sample', 	 dest="M", 		default = 5,   	type=int, 	help='Batch size')
	parser.add_argument('-gpu', 	 dest="gpu", 		default='1',			help='GPU to use')
	parser.add_argument('-name', 	 dest="name", 		default='test_'+str(uuid.uuid4()),help='Name of the run')
	parser.add_argument('-drop',	 dest="dropout", 	default=1.0,  	type=float,	help='Dropout for full connected layer')
	parser.add_argument('-rdrop',	 dest="rec_dropout", 	default=1.0,  	type=float,	help='Recurrent dropout for LSTM')
	parser.add_argument('-lr',	 dest="lr", 		default=0.0001,  type=float,	help='Learning rate')
	parser.add_argument('-lam_1',	 dest="lambda_1", 		default=0.5,  type=float,	help='transE weight')
	parser.add_argument('-lam_2',	 dest="lambda_2", 		default=0.25,  type=float,	help='entity loss weight')
	parser.add_argument('-margin', 	 dest="margin", 	default=1,   	type=float, 	help='margin')
	parser.add_argument('-batch', 	 dest="batch_size", 	default= 50000,   	type=int, 	help='Batch size')
	parser.add_argument('-epoch', 	 dest="max_epochs", 	default= 5000,   	type=int, 	help='Max epochs')
	parser.add_argument('-l2', 	 dest="l2", 		default=0.0, 	type=float, 	help='L2 regularization')
	parser.add_argument('-seed', 	 dest="seed", 		default=1234, 	type=int, 	help='Seed for randomization')
	parser.add_argument('-inp_dim',  dest="inp_dim", 	default = 128,   	type=int, 	help='Hidden state dimension of Bi-LSTM')
	parser.add_argument('-L1_flag',  dest="L1_flag", 	action='store_false',   	 	help='Hidden state dimension of FC layer')
	parser.add_argument('-onlytransE', dest="onlytransE", 	action='store_true', 		help='Evaluate model on only transE loss')

	parser.add_argument('-onlyTest', dest="onlyTest", 	action='store_true', 		help='Evaluate model for test data')
	parser.add_argument('-restore',	 dest="restore", 	action='store_true', 		help='Restore from the previous best saved model')
	parser.add_argument('-res_epoch',	     dest="restore_epoch", 	default=200,   type =int,		help='Restore from the previous best saved model')
	args = parser.parse_args()
	args.dataset = 'data/'+ args.data_type +'_'+ args.version+'/train.txt'
	args.entity2id = 'data/'+ args.data_type +'_'+ args.version+'/entity2id.txt'
	args.relation2id = 'data/'+ args.data_type +'_'+ args.version+'/relation2id.txt'
	args.valid_data  =  'data/'+ args.data_type +'_'+ args.version+'/valid.txt'
	args.test_data  =  'data/'+ args.data_type +'_'+ args.version+'/test.txt'
	args.triple2id  =   'data/'+ args.data_type +'_'+ args.version+'/triple2id.txt'
	# if not args.restore: args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")
	tf.set_random_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)
	Helper.set_gpu(args.gpu)
	model  = HyTE(args)
	print('model object created')
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		print('enter fitting')
		model.fit(sess)
