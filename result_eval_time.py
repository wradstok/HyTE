from functools import partial
from typing import Dict, List
import numpy as np
import argparse
import helper as Helper


def original(start: int, end: int, rank_map: Dict[int, int]) -> List:
    """ Temporal evaluation metric as seen in the original HyTE code and paper."""
    ranks = []
    for time in range(start, end + 1):
        rank, _ = rank_map[time]
        ranks.append(rank)

    return np.min(np.array(ranks))


def approach_2(start: int, end: int, rank_map: Dict[int, int]) -> List:
    """ Approach 2: record sum of ranks of all quads in the interval. Divide by best possible sum. """
    ranks = []
    for time in range(start, end + 1):
        rank, _ = rank_map[time]
        ranks.append(rank)

    interval = end - start + 1
    penalty = interval * ((1 + interval) / 2)
    res = np.sum(ranks) / penalty
    return res


def approach_3(start: int, end: int, rank_map: Dict[int, int]) -> List:
    """ Approach 3: Calculate scores of all groups of length `interval`. Returns rank of original interval."""
    interval_size = end - start

    # Very naive (inefficient) implementation, but it will do fine for the number of time classes
    # we currently handle. Can be updated in the future to use a rolling sum.
    groups = []
    for time in range(0, len(rank_map) - interval_size):
        score_sum = 0
        for offset in range(0, interval_size):
            score_sum += rank_map[time + offset][1]
        groups.append((time, score_sum))

    # Convert to numpy array and sort according to score.
    groups = np.array(groups)
    groups = groups[groups[:, 1].argsort()]

    # Return rank of the original interval.
    rank = np.where(groups == start)[0][0]
    return 1 + rank


eval_metrics = {
    "original": partial(original),
    "2": partial(approach_2),
    "3": partial(approach_3),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Eval model outputs")
    parser.add_argument("-model", dest="model", required=True, help="Dataset to use")
    parser.add_argument(
        "-mode",
        dest="mode",
        required=True,
        choices=["valid", "test"],
        help="Run for validation or test set",
    )
    parser.add_argument(
        "-test_freq",
        dest="freq",
        required=True,
        type=int,
        help="what is to be predicted",
    )
    parser.add_argument(
        "-eval_metric",
        dest="eval_metric",
        required=True,
        choices=eval_metrics.keys(),
        help="Which temporal evaluation metric to apply.",
    )
    return parser.parse_args()


args = parse_args()
print(f"Evaluating {args.model} in {args.mode} mode")

for k in range(args.freq, 30000, args.freq):
    try:
        true_output = open(f"temp_scope/{args.model}/{args.mode}.txt")
        model_time = open(f"temp_scope/{args.model}/{args.mode}_time_pred_{k}.txt")
    except FileNotFoundError:
        # Ran through all files, exiting.
        break

    predictions = Helper.parse_score_file(model_time)

    # Now, the index is time. Map the (score, time) tuples to time: (rank, score).
    time_rank_map = {}
    for i, row in enumerate(predictions):
        # Parse ranks. Increase ranks by 1 to make them 1-based instead of 0-based.
        rank_score = zip(range(1, len(row) + 1), row[:, 0])
        times = map(int, row[:, 1])
        time_rank_map[i] = dict(zip(times, rank_score))

    # Switch depending on evaluation metric used.
    ranks = []
    for i, row in enumerate(true_output):
        start, end = list(map(int, row.split()))  # Kinda hacky, but works.

        ranks.append(eval_metrics[args.eval_metric](start, end, time_rank_map[i]))

    ranks = np.array(ranks)

    print(f"Epoch {k} : MR {np.mean(ranks)}")
    print(f"Epoch {k} : MRR {np.mean(np.reciprocal(ranks))}")

    # Calculate & print hits@x
    for hit in [1, 3, 10]:
        hits_x = len(ranks[np.where(ranks <= hit)]) / float(len(ranks))
        print(f"Epoch {k} : HITS@{hit} {hits_x}")
