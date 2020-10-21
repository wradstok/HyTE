import random as random
from collections import defaultdict as ddict
import numpy as np


def tdns(self, head, rel, tail, triple_set):
    # Create a dictionary containing all possible heads/tails for each (head, rel) and (tail, rel) combination?
    keep_tail = ddict(set)
    keep_head = ddict(set)
    for z in range(len(head)):
        tup = (int(head[z]), int(rel[z]), int(self.start_idx[z]))
        keep_tail[tup].add(tail[z])
        tup = (int(tail[z]), int(rel[z]), int(self.start_idx[z]))
        keep_head[tup].add(head[z])

    max_time_class = len(self.year2id.keys())
    self.ph, self.pt, self.r, self.nh, self.nt, self.triple_time = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for triple_id in range(len(head)):
        neg_set, neg_time_set = set(), set()
        neg_time_set.add((tail[triple_id], rel[triple_id], self.start_idx[triple_id], head[triple_id],))
        sample_time = np.arange(max_time_class)
        head_corrupt = []
        random.shuffle(sample_time)
        possible_head = random.randint(0, self.max_ent - 1)  # Generate a random entity?

        # Loop over all time periods.
        for z, time in enumerate(sample_time):
            if time == self.start_idx[triple_id]:
                continue
            if (tail[triple_id], rel[triple_id], time) not in keep_head:
                continue
            for value in keep_head[(tail[triple_id], rel[triple_id], time)]:
                if value != head[triple_id] and (value, rel[triple_id], tail[triple_id]) not in neg_set:
                    if (tail[triple_id], rel[triple_id], self.start_idx[triple_id]) in keep_tail and value in keep_tail[
                        (tail[triple_id], rel[triple_id], self.start_idx[triple_id])
                    ]:
                        continue
                    head_corrupt.append(value)

        head_corrupt = list(set(head_corrupt))
        random.shuffle(head_corrupt)
        # Generate corruptions?

        for k in range(self.p.M):
            if k < len(head_corrupt):
                self.nh.append(head_corrupt[k])
                self.nt.append(tail[triple_id])
                self.r.append(rel[triple_id])
                self.ph.append(head[triple_id])
                self.pt.append(tail[triple_id])
                self.triple_time.append(self.start_idx[triple_id])
                neg_set.add((head_corrupt[k], rel[triple_id], tail[triple_id]))
            else:
                while (possible_head, rel[triple_id], tail[triple_id]) in triple_set or (
                    possible_head,
                    rel[triple_id],
                    tail[triple_id],
                ) in neg_set:
                    possible_head = random.randint(0, self.max_ent - 1)
                self.nh.append(possible_head)
                self.nt.append(tail[triple_id])
                self.r.append(rel[triple_id])
                self.ph.append(head[triple_id])
                self.pt.append(tail[triple_id])
                self.triple_time.append(self.start_idx[triple_id])
                neg_set.add((possible_head, rel[triple_id], tail[triple_id]))

    for triple in range(len(tail)):
        neg_set = set()
        neg_time_set = set()
        neg_time_set.add((head[triple], rel[triple], self.start_idx[triple], tail[triple]))
        sample_time = np.arange(max_time_class)
        random.shuffle(sample_time)
        tail_corrupt = []

        possible_tail = random.randint(0, self.max_ent - 1)
        for z, time in enumerate(sample_time):
            if time == self.start_idx[triple]:
                continue
            if (head[triple], rel[triple], time) not in keep_tail:
                continue
            for s, value in enumerate(keep_tail[(head[triple], rel[triple], time)]):
                if value != tail[triple] and (head[triple], rel[triple], value) not in neg_set:
                    if (head[triple], rel[triple], self.start_idx[triple]) in keep_head and value in keep_head[
                        (head[triple], rel[triple], self.start_idx[triple])
                    ]:
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
                neg_set.add((head[triple], rel[triple], tail_corrupt[k]))
            else:
                while (head[triple], rel[triple], possible_tail) in triple_set or (
                    head[triple],
                    rel[triple],
                    possible_tail,
                ) in neg_set:
                    possible_tail = random.randint(0, self.max_ent - 1)
                self.nh.append(head[triple])
                self.nt.append(possible_tail)
                self.r.append(rel[triple])
                self.ph.append(head[triple])
                self.pt.append(tail[triple])
                self.triple_time.append(self.start_idx[triple])
                neg_set.add((head[triple], rel[triple], possible_tail))

    self.max_time = len(self.year2id.keys())
    self.time_steps = sorted(self.year2id.values())
    self.data = list(zip(self.ph, self.pt, self.r, self.nh, self.nt, self.triple_time))
    self.data = self.data + self.data[0 : self.p.batch_size]


def tans(self, head, rel, tail, triple_set):
    self.ph, self.pt, self.r, self.nh, self.nt, self.triple_time = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for triple in range(len(head)):
        neg_set = set()
        for k in range(self.p.M):
            possible_head = random.randint(0, self.max_ent - 1)
            while (possible_head, rel[triple], tail[triple]) in triple_set or (
                possible_head,
                rel[triple],
                tail[triple],
            ) in neg_set:
                possible_head = random.randint(0, self.max_ent - 1)
            self.nh.append(possible_head)
            self.nt.append(tail[triple])
            self.r.append(rel[triple])
            self.ph.append(head[triple])
            self.pt.append(tail[triple])
            self.triple_time.append(self.start_idx[triple])
            neg_set.add((possible_head, rel[triple], tail[triple]))

    for triple in range(len(tail)):
        neg_set = set()
        for k in range(self.p.M):
            possible_tail = random.randint(0, self.max_ent - 1)
            while (head[triple], rel[triple], possible_tail) in triple_set or (
                head[triple],
                rel[triple],
                possible_tail,
            ) in neg_set:
                possible_tail = random.randint(0, self.max_ent - 1)
            self.nh.append(head[triple])
            self.nt.append(possible_tail)
            self.r.append(rel[triple])
            self.ph.append(head[triple])
            self.pt.append(tail[triple])
            self.triple_time.append(self.start_idx[triple])
            neg_set.add((head[triple], rel[triple], possible_tail))

    self.max_time = len(self.year2id.keys())
    self.data = list(zip(self.ph, self.pt, self.r, self.nh, self.nt, self.triple_time))
    self.data = self.data + self.data[0 : self.p.batch_size]


def timestamp(self, train_triples):
    # Turn train triples into individual head, relation, tail list (two times)
    posh, rela, post = map(list, zip(*train_triples))
    head, rel, tail = map(list, zip(*train_triples))

    # Append a copy of the triple or every class in which it appears (?).
    for i in range(len(posh)):
        if self.start_idx[i] < self.end_idx[i]:
            for j in range(self.start_idx[i] + 1, self.end_idx[i] + 1):
                head.append(posh[i])
                rel.append(rela[i])
                tail.append(post[i])
                self.start_idx.append(j)

    return head, rel, tail
