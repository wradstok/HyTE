import helper as Helper
import numpy as np
import tensorflow as tf

def common_train(self):
	""" Training function shared between both link prediction and temporal prediction. """
	pos_h_e = tf.squeeze(tf.nn.embedding_lookup(self.ent_embeddings, self.pos_head))
	pos_t_e = tf.squeeze(tf.nn.embedding_lookup(self.ent_embeddings, self.pos_tail))
	pos_r_e = tf.squeeze(tf.nn.embedding_lookup(self.rel_embeddings, self.rel))
	return pos_h_e, pos_t_e, pos_r_e

def test_link_prediction(self):
	""" Test the link prediction performance of the model. If self.pred-mode == 1 we test 
		head prediction performance. If self.pred_mode == -1 we test tail prediction perf."""
	def f_head():
		e2 = tf.squeeze(tf.nn.embedding_lookup(self.ent_embeddings, self.pos_tail))
		pos_h_e = self.ent_embeddings
		pos_t_e = tf.reshape(tf.tile(e2,[self.max_ent]),(self.max_ent, self.p.inp_dim))
		return pos_h_e, pos_t_e
		
	def f_tail():
		e1 = tf.squeeze(tf.nn.embedding_lookup(self.ent_embeddings, self.pos_head))
		pos_h_e = tf.reshape(tf.tile(e1,[self.max_ent]),(self.max_ent, self.p.inp_dim))
		pos_t_e = self.ent_embeddings
		return pos_h_e, pos_t_e

	##  pred_mode = 1 for head -1 for tail
	neutral = tf.constant(0)
	pos_h_e, pos_t_e  = tf.cond(self.pred_mode > neutral, f_head, f_tail)
	r  = tf.squeeze(tf.nn.embedding_lookup(self.rel_embeddings,self.rel))
	pos_r_e = tf.reshape(tf.tile(r,[self.max_ent]),(self.max_ent,self.p.inp_dim))
	return pos_h_e, pos_t_e, pos_r_e

def test_temp_prediction(self):	
	""" Test temporal prediction of the model. """
	e1 = tf.squeeze(tf.nn.embedding_lookup(self.ent_embeddings, self.pos_head))
	pos_h_e = tf.reshape(tf.tile(e1,[self.max_time]),(self.max_time, self.p.inp_dim))

	e2 = tf.squeeze(tf.nn.embedding_lookup(self.ent_embeddings, self.pos_tail))
	pos_t_e = tf.reshape(tf.tile(e2,[self.max_time]),(self.max_time, self.p.inp_dim))

	r  = tf.squeeze(tf.nn.embedding_lookup(self.rel_embeddings,self.rel))
	pos_r_e = tf.reshape(tf.tile(r,[self.max_time]),(self.max_time,self.p.inp_dim))
	return pos_h_e, pos_t_e, pos_r_e

def calculated_score_for_positive_elements(self, sess, quintuple, epoch, f_valid, eval_mode='valid'):
    start_lbl, end_lbl = Helper.get_span_ids(quintuple[3], quintuple[4], self.year2id)

    if eval_mode == 'test' or (eval_mode == "valid" and epoch == self.p.test_freq):
        f_valid.write(f'{quintuple[0]}\t{quintuple[1]}\t{quintuple[2]}\n')

    pos_head = sess.run(self.pos ,feed_dict = { self.pos_head:  	np.array([quintuple[0]]).reshape(-1,1), 
                                                self.rel:       	np.array([quintuple[1]]).reshape(-1,1), 
                                                self.pos_tail:	np.array([quintuple[2]]).reshape(-1,1),
                                                self.start_year :np.array([start_lbl]*self.max_ent),
                                                self.end_year : np.array([end_lbl]*self.max_ent),
                                                self.mode: 			   -1,
                                                self.pred_mode: 1,
                                                self.query_mode: 1})
    pos_head = np.squeeze(pos_head)
    
    pos_tail = sess.run(self.pos ,feed_dict = {    self.pos_head:  	np.array([quintuple[0]]).reshape(-1,1), 
                                                    self.rel:       	np.array([quintuple[1]]).reshape(-1,1), 
                                                    self.pos_tail:	np.array([quintuple[2]]).reshape(-1,1),
                                                    self.start_year :np.array([start_lbl]*self.max_ent),
                                                    self.end_year : np.array([end_lbl]*self.max_ent),
                                                    self.mode: 			   -1, 
                                                    self.pred_mode:  -1,
                                                    self.query_mode:  1})
    pos_tail = np.squeeze(pos_tail)


    pos_rel = sess.run(self.pos ,feed_dict = {    self.pos_head:  	np.array([quintuple[0]]).reshape(-1,1), 
                                                    self.rel:      	np.array([quintuple[1]]).reshape(-1,1), 
                                                    self.pos_tail:	np.array([quintuple[2]]).reshape(-1,1),
                                                    self.start_year:np.array([start_lbl]*self.max_rel),
                                                    self.end_year : np.array([end_lbl]*self.max_rel),
                                                    self.mode: 	 	-1, 
                                                    self.pred_mode: -1,
                                                    self.query_mode: -1})
    pos_rel = np.squeeze(pos_rel)

    return pos_head, pos_tail, pos_rel

def temp_test_against(self, sess, data, mode: str, epoch: int):
	if mode not in ['valid', 'test']:
		raise ValueError

	save_dir_results = Helper.get_temp_result_dir(self.p.name)
	
	# Open files
	if epoch == self.p.test_freq:
		f_valid = open(f'{save_dir_results}/{mode}.txt','w')
	f_time = open(f'{save_dir_results}/{mode}_time_pred_{epoch}.txt', 'w')

	# Run predictions
	for i, quintuple in enumerate(data):
		start_lbl, end_lbl = Helper.get_span_ids(quintuple[3], quintuple[4], self.year2id)

		if epoch == self.p.test_freq:
			f_valid.write(str(start_lbl)+'\t'+str(end_lbl)+'\n')

		pos_time = sess.run(self.pos, feed_dict = { self.pos_head:  	np.array([quintuple[0]]).reshape(-1,1), 
													self.rel:       	np.array([quintuple[1]]).reshape(-1,1), 
													self.pos_tail:	    np.array([quintuple[2]]).reshape(-1,1),
													self.start_year:    np.array(sorted(self.year2id.values())),
													self.end_year:      np.array([end_lbl]*self.max_ent),
                                                    self.mode: 	 	-1, 
                                                    self.query_mode: 1
												  }
							)


		pos_time = np.squeeze(pos_time)
		f_time.write(' '.join([str(x) for x in pos_time]) + '\n')
		
		if i % 1000 == 0:
			print(f'{i}. no of {mode} triples complete')

	# Close files.
	if epoch == self.p.test_freq:
		f_valid.close()
	f_time.close()

def test_link_against(self, sess, data, mode: str, epoch: int):
	if mode not in ['valid', 'test']:
		raise ValueError
	
	save_dir_results = Helper.get_result_dir(self.p.name)

	if epoch == self.p.test_freq:
		f_valid  = open(f'{save_dir_results }/{mode}.txt','w')
				
		fileout_head = open(f'{save_dir_results}/{mode}_head_pred_{epoch}.txt','w')
		fileout_tail = open(f'{save_dir_results}/{mode}_tail_pred_{epoch}.txt','w')
		fileout_rel  = open(f'{save_dir_results}/{mode}_rel_pred_{epoch}.txt', 'w')

		for i,t in enumerate(data):
			score = calculated_score_for_positive_elements(t, sess, epoch, f_valid, mode)
			if score:
				fileout_head.write(' '.join([str(x) for x in score[0]]) + '\n')
				fileout_tail.write(' '.join([str(x) for x in score[1]]) + '\n')
				fileout_rel.write (' '.join([str(x) for x in score[2]] ) + '\n')
	
			if i%500 == 0:
				print(f'{i}. no of {mode}_triples complete')

		fileout_head.close()
		fileout_tail.close()
		fileout_rel.close()

		if epoch == self.p.test_freq:
			f_valid.close()
		print("Validation Ended")