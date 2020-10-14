import argparse
import os
import pdb
import random
import uuid
from collections import defaultdict as ddict, Counter
from random import *

import pandas as pd
import scipy.sparse as sp
import tensorflow as tf

from helper import *
from models import *

YEARMIN = -50
YEARMAX = 3000
class HyTE(Model):

	def parse_quintuple(self, line: str):
		""" Read a (head, relation, tail, begin_time, end_time) quintuple from the dataset. 
			Time information is given at year level."""
		h, r, t, b, e = map(str.strip, line.split())
		b, e = b.split('-')[0], e.split("-")[0]
		return (h, r, t, b, e)

	def read_data(self,filename):
		valid_triples = []
		with open(filename,'r') as filein:
			for line in filein:
				valid_triples.append(list(self.parse_quintuple(line)))

		return valid_triples

	def getBatches(self, data, shuffle = True):
		if shuffle: 
			random.shuffle(data)

		num_batches = len(data) // self.p.batch_size

		for i in range(num_batches):
			start_idx = i * self.p.batch_size
			yield data[start_idx : start_idx + self.p.batch_size]


	def create_year2id(self,triple_time):
		year2id = dict()
		freq = ddict(int)
		count = 0
		year_list = []
		
		# Create a list of all years
		for triple_scope in triple_time.values():
			for year in triple_scope:
				# NOTE: This implementation ignores years consist of fewer than 4 digits. 
				if year.find("#") == -1 and len(year) == 4:
					year_list.append(int(year))

		# Count the number of occurrences for each year.
		freq = Counter(year_list)

		# Create classes.
		# NOTE: This implementation ignores the valid time of facts when generating classes.
		#		E.g., a fact with scope [1250,2000] is only counted as an occurence for 1250 and 2000.
		year_class = [] # Stores the final year of each class
		count = 0
		for key in sorted(freq.keys()):
			count += freq[key]
			if count > 300:
				year_class.append(key)
				count = 0

		# Create a dict of (begin, end) keys that have an ID as value.
		prev_year, i = 0, 0
		for i, end_year in enumerate(year_class):
			year2id[(prev_year, end_year)] = i
			prev_year = end_year + 1

		year2id[(prev_year, max(year_list))] = i + 1
		self.year_list = sorted(year_list)
		return year2id
	
	def get_span_ids(self, start, end):
		start =int(start)
		end=int(end)
		if start > end:
			end = YEARMAX

		if start == YEARMIN:
			start_lbl = 0
		else:
			for key,lbl in sorted(self.year2id.items(), key=lambda x:x[1]):
				if start >= key[0] and start <= key[1]:
					start_lbl = lbl
		
		if end == YEARMAX:
			end_lbl = len(self.year2id.keys())-1
		else:
			for key,lbl in sorted(self.year2id.items(), key=lambda x:x[1]):
				if end >= key[0] and end <= key[1]:
					end_lbl = lbl
		return start_lbl, end_lbl

	def create_id_labels(self,triple_time):
		YEARMAX = 3000
		YEARMIN =  -50
		
		# Input_idx stores the ID's of triples that passed validation of temporal information.
		# Start_idx stores temporal class indices?
		# End_idx stores temporal class indices?
		inp_idx, start_idx, end_idx = [], [], []
		
		for triple_idx, triple_scope in triple_time.items():
			# Again, parse the triple scopes like in year2id.
			# Ignores years that are not 4 digits long.
			start = triple_scope[0].split('-')[0]
			end = triple_scope[1].split('-')[0]
			if start == '####':
				start = YEARMIN
			elif start.find('#') != -1 or len(start) != 4:
				continue

			if end == '####':
				end = YEARMAX
			elif end.find('#') != -1 or len(end)!=4:
				continue
			
			start = int(start)
			end   = int(end)
			
			if start > end:
				end = YEARMAX

			inp_idx.append(triple_idx)

			# Find the index of the class in which this triple starts.
			if start == YEARMIN:
				start_idx.append(0)
			else:
				for key, lbl in self.year2id.items():
					if start >= key[0] and start <= key[1]:
						start_idx.append(lbl)
						break
			
			# Find the index of the class in which this triple ends.
			if end == YEARMAX:
				end_idx.append(len(self.year2id.keys())-1)
			else:
				for key,lbl in self.year2id.items():
					if end >= key[0] and end <= key[1]:
						end_idx.append(lbl)
						break

		return inp_idx, start_idx, end_idx

	def load_data(self):
		triple_set = set()
		with open(self.p.triple2id,'r') as filein:
			for line in filein:
				h, r, t, _, _ = self.parse_quintuple(line)
				triple_set.add((h,r,t))

		train_triples = []
		self.start_time , self.end_time, self.num_class  = ddict(dict), ddict(dict), ddict(dict)
		triple_time = dict()
		self.inp_idx, self.start_idx, self.end_idx ,self.labels = ddict(list), ddict(list), ddict(list), ddict(list)

		# Load training data.
		with open(self.p.dataset,'r') as train_data:
			for i, line in enumerate(train_data):
				h, r, t, b, e = self.parse_quintuple(line)
				train_triples.append([h,r,t])
				triple_time[i] = [b, e]

		# Count # entities
		self.max_ent = 0
		with open(self.p.entity2id,'r', encoding="utf8") as entity_data:
			for line in entity_data:
				self.max_ent += 1

		# Count # relations
		self.max_rel = 0
		with open(self.p.relation2id, 'r') as filein3:
			for line in filein3:
				self.max_rel += 1

		# Create year classes, e.g. (0-1853)
		self.year2id = self.create_year2id(triple_time)
		self.num_class = len(self.year2id.keys())

		self.inp_idx['triple'], self.start_idx, self.end_idx = self.create_id_labels(triple_time)

		# Delete all triples from train whose scope could not be parsed in create_id_labels
		# As these do not have an entry in inp_idx['triple']
		keep_idx = set(self.inp_idx['triple'])
		for i in range (len(train_triples)-1,-1,-1):
			if i not in keep_idx:
				del train_triples[i]

		# Turn train triples into individual head, relation, tail list (two times)		
		posh, rela, post =  map(list, zip(*train_triples))
		head, rel, tail = map(list, zip(*train_triples))

		# Append a copy of the triple or every class in which it appears.
		for i in range(len(posh)):
			if self.start_idx[i] < self.end_idx[i]:
				for j in range(self.start_idx[i] + 1, self.end_idx[i] + 1):
					head.append(posh[i])
					rel.append(rela[i])
					tail.append(post[i])
					self.start_idx.append(j)

		# Create a dictionary containing all possible heads/tails for each (head, rel) and (tail, rel) combination?
		keep_tail = ddict(set)
		keep_head = ddict(set)
		for z in range(len(head)):
			tup = (int(head[z]), int(rel[z]), int(self.start_idx[z]))
			keep_tail[tup].add(tail[z])
			tup = (int(tail[z]), int(rel[z]), int(self.start_idx[z]))
			keep_head[tup].add(head[z])

		# I give up.
		max_time_class = len(self.year2id.keys())
		self.ph, self.pt, self.r,self.nh, self.nt , self.triple_time  = [], [], [], [], [], []
		for triple_id in range(len(head)):
			neg_set, neg_time_set = set(), set()
			neg_time_set.add((tail[triple_id], rel[triple_id], self.start_idx[triple_id], head[triple_id]))
			sample_time = np.arange(max_time_class)
			head_corrupt = []
			random.shuffle(sample_time)
			possible_head = randint(0, self.max_ent - 1) # Generate a random entity?

			for z, time in enumerate(sample_time):
				if time == self.start_idx[triple_id]: 
					continue
				if (tail[triple_id], rel[triple_id], time) not in keep_head: 
					continue
				for value in keep_head[(tail[triple_id], rel[triple_id], time)]:
					if value != head[triple_id] and (value, rel[triple_id], tail[triple_id]) not in neg_set:
						if (tail[triple_id], rel[triple_id], self.start_idx[triple_id]) in keep_tail and value in keep_tail[(tail[triple_id], rel[triple_id], self.start_idx[triple_id])]:
							continue
						head_corrupt.append(value)

			head_corrupt = list(set(head_corrupt))
			random.shuffle(head_corrupt)
			for k in range(self.p.M):
				if k < len(head_corrupt):
					self.nh.append(head_corrupt[k])
					self.nt.append(tail[triple_id])
					self.r.append(rel[triple_id])
					self.ph.append(head[triple_id])
					self.pt.append(tail[triple_id])
					self.triple_time.append(self.start_idx[triple_id])
					neg_set.add((head_corrupt[k], rel[triple_id],tail[triple_id]))
				else:
					# break
					while (possible_head, rel[triple_id], tail[triple_id]) in triple_set or (possible_head, rel[triple_id],tail[triple_id]) in neg_set:
						possible_head = randint(0,self.max_ent-1)
					self.nh.append(possible_head)
					self.nt.append(tail[triple_id])
					self.r.append(rel[triple_id])
					self.ph.append(head[triple_id])
					self.pt.append(tail[triple_id])
					self.triple_time.append(self.start_idx[triple_id])
					neg_set.add((possible_head, rel[triple_id],tail[triple_id]))
		
		for triple in range(len(tail)):
			neg_set = set()
			neg_time_set = set()
			neg_time_set.add((head[triple], rel[triple], self.start_idx[triple], tail[triple]))
			sample_time = np.arange(max_time_class)
			random.shuffle(sample_time)
			tail_corrupt = []
			
			possible_tail = randint(0,self.max_ent-1)
			for z, time in enumerate(sample_time):
				if time == self.start_idx[triple]: continue
				if (head[triple], rel[triple], time) not in keep_tail: continue
				for s, value in enumerate(keep_tail[(head[triple], rel[triple], time)]):
					if value != tail[triple] and (head[triple], rel[triple], value) not in neg_set:
						if (head[triple], rel[triple], self.start_idx[triple]) in keep_head  and value in keep_head[(head[triple], rel[triple], self.start_idx[triple])]:
							continue
						tail_corrupt.append(value)
			tail_corrupt = list(set(tail_corrupt))
			random.shuffle(tail_corrupt)
			for k in range(self.p.M):
				if k < len(tail_corrupt):
					self.nh.append(head[triple])
					self.nt.append(tail_corrupt[k])
					self.r.append(rel[triple])
					self.ph.append(head[triple])
					self.pt.append(tail[triple])
					self.triple_time.append(self.start_idx[triple])
					neg_set.add((head[triple], rel[triple],tail_corrupt[k]))
				else:
					while (head[triple], rel[triple],possible_tail) in triple_set or (head[triple], rel[triple],possible_tail) in neg_set:
							possible_tail = randint(0,self.max_ent - 1)
					self.nh.append(head[triple])
					self.nt.append(possible_tail)
					self.r.append(rel[triple])
					self.ph.append(head[triple])
					self.pt.append(tail[triple])
					self.triple_time.append(self.start_idx[triple])
					neg_set.add((head[triple], rel[triple],possible_tail))

		self.max_time = len(self.year2id.keys())
		self.time_steps = sorted(self.year2id.values())
		self.data = list(zip(self.ph, self.pt, self.r , self.nh, self.nt, self.triple_time))
		self.data = self.data + self.data[0:self.p.batch_size]
		for i in range(len(self.ph)):
			if self.ph[i] == self.nh[i] and self.pt[i] == self.nt[i]:
				print("False")

	def add_placeholders(self):
		self.start_year = tf.placeholder(tf.int32 ,shape=[None], name = 'start_time')
		self.end_year   = tf.placeholder(tf.int32 ,shape=[None],name = 'end_time')
		self.pos_head 	= tf.placeholder(tf.int32, [None,1])
		self.pos_tail 	= tf.placeholder(tf.int32, [None,1])
		self.rel      	= tf.placeholder(tf.int32, [None,1])
		self.neg_head 	= tf.placeholder(tf.int32, [None,1])
		self.neg_tail 	= tf.placeholder(tf.int32, [None,1])
		self.mode 	  	= tf.placeholder(tf.int32,shape = ())
		self.dropout 		= tf.placeholder_with_default(self.p.dropout, 	  shape=(), name='dropout')
		self.rec_dropout 	= tf.placeholder_with_default(self.p.rec_dropout, shape=(), name='rec_dropout')

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
		else: 
			feed_dict[self.mode] = -1
			feed_dict[self.dropout]     = 1.0
			feed_dict[self.rec_dropout] = 1.0

		return feed_dict


	def time_projection(self,data,t):
		inner_prod  = tf.tile(tf.expand_dims(tf.reduce_sum(data*t,axis=1),axis=1),[1,self.p.inp_dim])
		prod 		= t*inner_prod
		data = data - prod
		return data

	def add_model(self):
		with tf.name_scope("embedding"):
			self.ent_embeddings =  tf.get_variable(name = "ent_embedding",  shape = [self.max_ent, self.p.inp_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = False), regularizer=self.regularizer)
			self.rel_embeddings =  tf.get_variable(name = "rel_embedding",  shape = [self.max_rel, self.p.inp_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = False), regularizer=self.regularizer)
			self.time_embeddings = tf.get_variable(name = "time_embedding",shape = [self.max_time, self.p.inp_dim], initializer = tf.contrib.layers.xavier_initializer(uniform =False))
		
			transE_in_dim = self.p.inp_dim
			transE_in     = self.ent_embeddings
		####################------------------------ time aware GCN MODEL ---------------------------##############


	
		## Some transE style model ####
		neutral = tf.constant(0)

		def f_train():
			pos_h_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_head))
			pos_t_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_tail))
			pos_r_e = tf.squeeze(tf.nn.embedding_lookup(self.rel_embeddings, self.rel))
			t_1 = tf.squeeze(tf.nn.embedding_lookup(self.time_embeddings, self.start_year))
			return pos_h_e, pos_t_e, pos_r_e, t_1
		def f_test():
			e1 = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_head))
			e2 = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_tail))
			pos_h_e = tf.reshape(tf.tile(e1,[self.max_time]),(self.max_time, transE_in_dim))
			pos_t_e = tf.reshape(tf.tile(e2,[self.max_time]),(self.max_time, transE_in_dim))
			r  = tf.squeeze(tf.nn.embedding_lookup(self.rel_embeddings,self.rel))
			pos_r_e = tf.reshape(tf.tile(r,[self.max_time]),(self.max_time,transE_in_dim))
			t_1 = tf.squeeze(tf.nn.embedding_lookup(self.time_embeddings, self.start_year))
			return pos_h_e, pos_t_e, pos_r_e, t_1

		pos_h_e, pos_t_e, pos_r_e, t_1 = tf.cond(self.mode > neutral, f_train, f_test)
		neg_h_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.neg_head))
		neg_t_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.neg_tail))

		#### ----- time -----###
		
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
			if self.regularizer != None: loss += tf.contrib.layers.apply_regularization(self.regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
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
		if self.p.l2 == 0.0: 	
			self.regularizer = None
		else:
			self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.p.l2)
		self.load_data()
		self.nbatches = len(self.data) // self.p.batch_size
		self.add_placeholders()
		self.pos, neg = self.add_model()
		self.loss      	= self.add_loss(self.pos, neg)
		self.train_op  	= self.add_optimizer(self.loss)
		self.merged_summ = tf.summary.merge_all()
		self.summ_writer = None
		print('model done')

	def run_epoch(self, sess,data, epoch):
		losses = []

		for _, batch in enumerate(self.getBatches(data, shuffle)):
			feed = self.create_feed_dict(batch)
			l, a = sess.run([self.loss, self.train_op], feed_dict = feed)
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
			save_path = os.path.join(save_dir, 'epoch_{}'.format(self.p.restore_epoch))
			saver.restore(sess, save_path)
		print('start fitting')
		self.best_prf = None
		validation_data = self.read_data(self.p.valid_data)

		if not self.p.onlyTest:
			for epoch in range(self.p.max_epochs):
				l = self.run_epoch(sess,self.data,epoch)
				
				if epoch%50 == 0:
					print(f'Epoch {epoch}\t Loss {l}\t model {self.p.name}')

				if epoch % self.p.test_freq == 0 and epoch != 0:
					## -- check pointing -- ##
					save_path = os.path.join(save_dir, 'epoch_{}'.format(epoch))   
					saver.save(sess=sess, save_path=save_path)
					save_dir_results = 'temp_scope/'+ self.p.name + '/'
					if not os.path.exists(save_dir_results): os.makedirs(save_dir_results) 
					
					if epoch == self.p.test_freq:
						f_valid = open(save_dir_results+'/valid.txt','w')
					f_time = open(save_dir_results+'/valid_time_pred_{}.txt'.format(epoch),'w')
					
					for i,t in enumerate(validation_data):
						loss =np.zeros(self.max_ent)
						start_trip 	= t[3][0].split('-')[0]
						end_trip 	= t[3][1].split('-')[0]
						if start_trip == '####':
							start_trip = YEARMIN
						elif start_trip.find('#') != -1 or len(start_trip)!=4:
							continue

						if end_trip == '####':
							end_trip = YEARMAX
						elif end_trip.find('#')!= -1 or len(end_trip)!=4:
							continue
							
						start_lbl, end_lbl = self.get_span_ids(start_trip, end_trip)
						if epoch == self.p.test_freq:
							f_valid.write(str(start_lbl)+'\t'+str(end_lbl)+'\n')
						pos_time = sess.run(self.pos ,feed_dict = { self.pos_head:  	np.array([t[0]]).reshape(-1,1), 
																   	self.rel:       	np.array([t[1]]).reshape(-1,1), 
																   	self.pos_tail:	    np.array([t[2]]).reshape(-1,1),
																   	self.start_year:    np.array(self.time_steps),
																   	self.end_year:      np.array([end_lbl]*self.max_ent),
																   	self.mode: 			-1,
																   }
										    )
		

						pos_time = np.squeeze(pos_time)
						f_time.write(' '.join([str(x) for x in pos_time]) + '\n')
						
						if i%1000 == 0:
							print('{}. no of valid_triples complete'.format(i))
						
					
					if epoch == self.p.test_freq:
						f_valid.close()
					f_time.close()
					print("Validation Ended")
		else: 
			time_mat = sess.run(self.time_embeddings)
			df = pd.DataFrame(data=time_mat.astype(float)) 
			df.to_csv(self.p.name + str(self.p.restore_epoch) + '.tsv', sep='\t', header=False, float_format='%.6f', index=False)

if __name__== "__main__":
	print('here in main')
	parser = argparse.ArgumentParser(description='KG temporal inference using GCN')

	parser.add_argument('-data_type', dest= "data_type", default ='yago', choices = ['yago','wiki_data'], help ='dataset to choose')
	parser.add_argument('-version',dest = 'version', default = 'large', help = 'data version to choose')
	parser.add_argument('-test_freq', 	 dest="test_freq", 	default = 25,   	type=int, 	help='Batch size')
	parser.add_argument('-neg_sample', 	 dest="M", 		default = 1,   	type=int, 	help='Batch size')
	parser.add_argument('-gpu', 	 dest="gpu", 		default='1',			help='GPU to use')
	parser.add_argument('-name', 	 dest="name", 		default='test_'+str(uuid.uuid4()),help='Name of the run')
	parser.add_argument('-embed', 	 dest="embed_init", 	default='wiki_300',	 	help='Embedding for initialization')
	parser.add_argument('-drop',	 dest="dropout", 	default=1.0,  	type=float,	help='Dropout for full connected layer')
	parser.add_argument('-rdrop',	 dest="rec_dropout", 	default=1.0,  	type=float,	help='Recurrent dropout for LSTM')
	parser.add_argument('-lr',	 dest="lr", 		default=0.0001,  type=float,	help='Learning rate')
	parser.add_argument('-margin', 	 dest="margin", 	default= 10 ,   	type=float, 	help='margin')
	parser.add_argument('-batch', 	 dest="batch_size", 	default= 50000,   	type=int, 	help='Batch size')
	parser.add_argument('-epoch', 	 dest="max_epochs", 	default= 5000,   	type=int, 	help='Max epochs')
	parser.add_argument('-l2', 	 dest="l2", 		default=0.0, 	type=float, 	help='L2 regularization')
	parser.add_argument('-seed', 	 dest="seed", 		default=1234, 	type=int, 	help='Seed for randomization')
	parser.add_argument('-inp_dim',  dest="inp_dim", 	default = 128,   	type=int, 	help='Hidden state dimension of Bi-LSTM')
	parser.add_argument('-L1_flag',  dest="L1_flag", 	action='store_false',   	 	help='Hidden state dimension of FC layer')
	parser.add_argument('-onlyTest', dest="onlyTest", 	action='store_true', 		help='Evaluate model on test')
	parser.add_argument('-onlytransE', dest="onlytransE", 	action='store_true', 		help='Evaluate model on only transE loss')
	parser.add_argument('-restore',	 		 dest="restore", 	    action='store_true', 		help='Restore from the previous best saved model')
	parser.add_argument('-res_epoch',	     dest="restore_epoch", 	default=200,   type =int,		help='Restore from the previous best saved model')
	
	args = parser.parse_args()
	args.dataset = 'data/'+ args.data_type +'_'+ args.version+'/train.txt'
	args.entity2id = 'data/'+ args.data_type +'_'+ args.version+'/entity2id.txt'
	args.relation2id = 'data/'+ args.data_type +'_'+ args.version+'/relation2id.txt'
	args.test_data  =  'data/'+ args.data_type +'_'+ args.version+'/test.txt'
	args.valid_data  =  'data/'+ args.data_type +'_'+ args.version+'/valid.txt'
	args.triple2id  =   'data/'+ args.data_type +'_'+ args.version+'/triple2id.txt'
	args.embed_dim = int(args.embed_init.split('_')[1])

	# if not args.restore: 
	# 	args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")
	tf.set_random_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)
	set_gpu(args.gpu)
	model  = HyTE(args)
	print('model object created')
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		print('enter fitting')
		model.fit(sess)
