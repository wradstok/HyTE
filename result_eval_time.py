import numpy as np
import argparse,pdb
parser = argparse.ArgumentParser(description='Eval model outputs')
parser.add_argument('-model', 	 	dest = "model", required=True,				help='Dataset to use')
parser.add_argument('-test_freq', 	dest = "freq", 	required=True,	type =int,  help='what is to be predicted')

args = parser.parse_args()

	
print(args.model)

for k in range(args.freq,30000,args.freq):
	try:
		valid_output = open('results/temp_scope/'+args.model+'/valid.txt')
		model_time = open('results/temp_scope/'+args.model+'/valid_time_pred_{}.txt'.format(k))
	except FileNotFoundError:
		# Ran through all files, exiting.	
		break

	model_out_time = []
	for line in model_time:
		scores = list(map(float, line.split()))
		scores = list(zip(scores, range(0, len(scores)))) # Add index
		scores = np.array(scores) # Convert to np_array
		model_out_time.append(scores[scores[:,0].argsort()]) # Sort according to score
	
	final_out_time = []
	for row in model_out_time:
	    temp_dict =dict()
	    count = 0
	    for ele in row:
	        temp_dict[ele[1]] = count
	        count += 1
	    final_out_time.append(temp_dict)
	
	ranks_time = []
	
	for i,row in enumerate(valid_output):
		avg_rank = []
		top_time = final_out_time[i][0]
		start_time = int(row.split()[0])
		end_time   = int(row.split()[1])
		for e in range(start_time,end_time + 1):
			avg_rank.append(final_out_time[i][e])
		ranks_time.append(np.min(np.array(avg_rank)))


	print('Epoch {} :  time_rank {}'.format(k ,np.mean(np.array(ranks_time))))
