from typing import List
import numpy as np
import argparse, sys
import helper as Helper


def parse_args():
    parser = argparse.ArgumentParser(description="Eval model outputs")
    parser.add_argument("-model", dest="model", required=True, help="Dataset to use")
    parser.add_argument(
        "-eval_mode",
        dest="eval_mode",
        required=True,
        help="To evaluate test or validation",
    )
    parser.add_argument(
        "-test_freq",
        dest="freq",
        required=True,
        type=int,
        help="what is to be predicted",
    )
    parser.parse_args()
    return args


# Initialize `best` values as maximum possible.
best_rank, best_epoch, best_tail_rank, best_head_rank = sys.maxsize

args = parse_args()
print(args.model)

for k in range(args.freq, 30000, args.freq):
    try:
        true_output = open(f"results/{args.model}/{args.eval_mode}.txt")
        model_output_head = open(
            f"results/{args.model}/{args.eval_mode}_head_pred_{k}.txt"
        )
        model_output_tail = open(
            f"results/{args.model}/{args.eval_mode}_tail_pred_{l}.txt"
        )
    except FileNotFoundError:
        # Ran through all files, exiting.
        break

    # Read the head scores
    model_out_head = Helper.parse_score_file(model_output_head)
    model_out_tail = Helper.parse_score_file(model_output_tail)

    # Find & record the rank of the original triple by comparing scores.
    ranks_head, ranks_tail = [], []
    for i, row in enumerate(true_output):
        row = list(map(float, row.split()))  # Kinda hacky, but works.
        # Double [0] is to find the actual index
        ranks_head.append(np.where(model_out_head[i][:, 1] == row[0])[0][0])
        ranks_tail.append(np.where(model_out_tail[i][:, 1] == row[2])[0][0])

    # Convert to numpy array
    ranks_tail = np.array(ranks_tail) + 1
    ranks_head = np.array(ranks_head) + 1

    # Calculate & print mr
    mr_tail = np.mean(ranks_tail)
    mr_head = np.mean(ranks_head)
    print(f"Epoch {k} : {args.eval_mode}_MR {(mr_tail + mr_head) / 2}")

    # Calculate & print mrr
    mrr_tail = np.mean(np.reciprocal(ranks_tail))
    mrr_head = np.mean(np.reciprocal(ranks_head))
    print(f"Epoch {k} : {args.eval_mode}_MRR {(mrr_tail + mrr_head / 2)}")

    # Calculate & print hits@x
    for hit in [1, 3, 10]:
        hits_x_tail = len(ranks_tail[np.where(ranks_tail <= hit)]) / float(
            len(ranks_tail)
        )
        hits_x_head = len(ranks_head[np.where(ranks_head <= hit)]) / float(
            len(ranks_head)
        )

        print(
            f"Epoch {k} : {args.eval_mode}_HITS@{hit} {(hits_x_tail + hits_x_head) / 2}"
        )

    if args.eval_mode == "valid":
        if (
            np.mean(np.array(ranks_tail)) + 1 + np.mean(np.array(ranks_head)) + 1
        ) / 2 < best_rank:
            best_rank = (
                np.mean(np.array(ranks_tail)) + 1 + np.mean(np.array(ranks_head)) + 1
            ) / 2
            best_epoch = k
            best_tail_rank = np.mean(np.array(ranks_tail)) + 1
            best_head_rank = np.mean(np.array(ranks_head)) + 1
        print("------------------------------------------")
        print(
            "Best Validation Epoch till now Epoch {}, tail rank: {}, head rank: {}".format(
                best_epoch, best_tail_rank, best_head_rank
            )
        )
        print("------------------------------------------")
