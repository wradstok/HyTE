from typing import List
import numpy as np, os
from collections import Counter
import io

np.set_printoptions(precision=4)

YEARMIN = -50
YEARMAX = 3000

def set_gpu(gpus):
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def parse_score_file(file: io.TextIOWrapper) -> List:
    """ Parse a scoring file. Outputs a list, where each entry is a numpy array containing
    (score, tripleid) tuples, sorted by score. """
    output = []
    for line in file:
        # Read scores as floats
        scores = list(map(float, line.split()))
        # Turn into (score, index) tuple
        scores = list(zip(scores, range(0, len(scores))))
        # Convert to numpy array
        scores = np.array(scores) 
        # Sort according to score. Now we have an array of (score, index) tuples.
        output.append(scores[scores[:,0].argsort()]) # Sort according to score
    return output

def get_line_count(filename: str) -> int:
    """ Get the number of lines in a file by reading it to the end."""
    count = 0
    with open(filename, 'r', encoding="utf-8") as file_in:
        for _ in file_in:
            count += 1
    return count

def get_span_ids(start: int, end: int, year2id):
    # Find the index of the class in which this triple starts.
    start_lbl, end_lbl = 0,  len(year2id.keys()) - 1

    # Find the index of the class in which this triple starts.
    # If it is min_year, none will match, and the label stays 0.
    for key, lbl in year2id.items():
        if start >= key[0] and start <= key[1]:
            start_lbl = lbl
            break
    
    # Find the index of the class in which this triple ends.
    # If it is max_year none will match, and it will stay the maximum.
    for key,lbl in year2id.items():
        if end >= key[0] and end <= key[1]:
            end_lbl = lbl
            break

    return start_lbl, end_lbl

def create_year2id(triple_time):
    year2id = dict()
    year_list = []
    
    # Create a list of all years, excluding the min_year and max_year
    for triple_scope in triple_time.values():
        for year in triple_scope:
            if year != YEARMIN and year != YEARMAX:
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
    return sorted(year_list), year2id

def create_id_labels(triple_time, year2id):
    # Input_idx stores the ID's of triples that passed validation of temporal information.
    inp_idx, start_idx, end_idx = [], [], []
    
    for triple_idx, triple_scope in triple_time.items():
        start = triple_scope[0]
        end = triple_scope[1]
    
        inp_idx.append(triple_idx)

        start_label, end_label = get_span_ids(start, end, year2id)
        start_idx.append(start_label)
        end_idx.append(end_label)

    return inp_idx, start_idx, end_idx

def parse_quintuple(line: str):
    """ Read a (head, relation, tail, begin_time, end_time) quintuple from the dataset. 
        Time information is given at year level. If time could not be parsed, -1 is returned
        as the begin and end time, and the user should discard the quintuple."""
    h, r, t, b, e = map(str.strip, line.split())
    b, e = b.split('-')[0], e.split("-")[0]

    # NOTE: This implementation ignores years consist of fewer than 4 digits. 

    # Try to parse begin time.
    if b == '####':
        b = YEARMIN
    elif b.find('#') != -1 or len(b) != 4:
        return (h,r,t,-1,-1)

    # Try to parse end time.
    if e == '####':
        e = YEARMAX
    elif e.find('#') != -1 or len(e)!=4:
        return (h,r,t,-1,-1)
            
    b = int(b)
    e = int(e)
            
    if b > e:
        e = YEARMAX
    return (h, r, t, b, e)

def read_data(filename: str):
    triples = []
    with open(filename, 'r') as filein:
        for line in filein:
            (h,r,t,b,e) = parse_quintuple(line)
            if b != -1 and e != -1:
                triples.append([h,r,t,b,e])

    return triples

def get_checkpoint_dir(model_name: str) -> str:
    dir = f'checkpoints/{model_name}/'
    create_dir_if_not_exists(dir)
    return dir

def get_result_dir(model_name: str) -> str:
    dir = f'results/{model_name}/'
    create_dir_if_not_exists(dir)
    return dir

def get_temp_result_dir(model_name: str) -> str:
    dir = f'temp_scope/{model_name}/'
    create_dir_if_not_exists(dir)
    return dir

def create_dir_if_not_exists(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)