#!/usr/bin/env python3

import time
import argparse

import numpy as np

import utils as u
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

VERBOSE = False

BINS = 5
K = None

# LOCAL_ALPHA has an effect on execution time. Too strict alpha will produce a sparse graph
# so we might need to run phase-1 multiple times to get up to k elements. Too relaxed alpha
# will give dense graph so the size of the separating set will increase and phase-1 will
# take more time.
# We tried a few different values and found that 0.01 gives the best result in our case
# (between 0.001 and 0.1).
LOCAL_ALPHA = 0.001
DEFAULT_GAMMA = 3

# original: 0.01 and 5

# SRC_DIR = 'sock-shop-data/carts-mem/1/'
# SRC_DIR = 'data/s-2/n-10-d-3-an-1-nor-s-1000-an-s-1000/'
SRC_DIR = 'dataset/'

# Split the dataset into multiple subsets
def create_chunks(df, gamma):
    chunks = list()
    names = np.random.permutation(df.columns)
    for i in range(df.shape[1] // gamma + 1):
        chunks.append(names[i * gamma:(i * gamma) + gamma])

    if len(chunks[-1]) == 0:
        chunks.pop()
    return chunks

# def run_level(normal_df, anomalous_df, gamma, localized, bins, verbose, knowledge=None):
#     ci_tests = 0
#     chunks = create_chunks(normal_df, gamma)
#     if verbose:
#         print(f"Created {len(chunks)} subsets")
#     if verbose: print(chunks)

#     f_child_union = list()
#     mi_union = list()
#     f_child = list()
#     for c in chunks:
#         # Try this segment with multiple values of alpha until we find at least one node
#         rc, _, mi, ci = u.top_k_rc(normal_df.loc[:, c],
#                                    anomalous_df.loc[:, c],
#                                    bins=bins,
#                                    localized=localized,
#                                    start_alpha=LOCAL_ALPHA,
#                                    min_nodes=1,
#                                    verbose=verbose,
#                                    knowledge = knowledge)
#         f_child_union += rc
#         mi_union += mi
#         ci_tests += ci
#         if verbose:
#             f_child.append(rc)

#     if verbose:
#         print(f"Output of individual chunk {f_child}")
#         print(f"Total nodes in mi => {len(mi_union)} | {mi_union}")

#     return f_child_union, mi_union, ci_tests

def run_level(normal_df, anomalous_df, gamma, localized, bins, verbose, knowledge=None):
    ci_tests = 0
    chunks = create_chunks(normal_df, gamma)
    if verbose:
        print(f"Created {len(chunks)} subsets")
    if verbose: print(chunks)

    f_child_union = list()
    mi_union = list()
    f_child = list()
    for c in chunks:
        # Filter the knowledge based on the current chunk
        if knowledge is not None:
            filtered_knowledge = [pair for pair in knowledge if pair[0] in c and pair[1] in c]
        else:
            filtered_knowledge = None

        rc, _, mi, ci = u.top_k_rc(normal_df.loc[:, c],
                                   anomalous_df.loc[:, c],
                                   bins=bins,
                                   localized=localized,
                                   start_alpha=LOCAL_ALPHA,
                                   min_nodes=1,
                                   verbose=verbose,
                                   knowledge=filtered_knowledge)  # Use filtered knowledge
        f_child_union += rc
        mi_union += mi
        ci_tests += ci
        if verbose:
            f_child.append(rc)

    if verbose:
        print(f"Output of individual chunk {f_child}")
        print(f"Total nodes in mi => {len(mi_union)} | {mi_union}")

    return f_child_union, mi_union, ci_tests


def run_multi_phase(normal_df, anomalous_df, gamma, localized, bins, verbose, knowledge = None):
    f_child_union = normal_df.columns
    mi_union = []
    i = 0
    prev = len(f_child_union)

    # Phase-1
    while True:
        start = time.time()
        f_child_union, mi, ci_tests = run_level(normal_df.loc[:, f_child_union],
                                                anomalous_df.loc[:, f_child_union],
                                                gamma, localized, bins, verbose , knowledge)
        if verbose:
            print(f"Level-{i}: variables {len(f_child_union)} | time {time.time() - start}")
        i += 1
        mi_union += mi
        # Phase-1 with only one level
        # break

        len_child = len(f_child_union)
        # If found gamma nodes or if running the current level did not remove any node
        if len_child <= gamma or len_child == prev: break
        prev = len(f_child_union)

    # Phase-2
    mi_union = []
    new_nodes = f_child_union
    # print(len(new_nodes))
    if knowledge is not None:
        filtered_knowledge = [pair for pair in knowledge if pair[0] in new_nodes and pair[1] in new_nodes]
        # print(filtered_knowledge)
    else:
        filtered_knowledge = None

    rc, _, mi, ci = u.top_k_rc(normal_df.loc[:, new_nodes],
                               anomalous_df.loc[:, new_nodes],
                               bins=bins,
                               mi=mi_union,
                               localized=localized,
                               verbose=verbose,
                               knowledge = filtered_knowledge)
    ci_tests += ci

    return rc, ci_tests

def rca_with_rcd(normal_df, anomalous_df, bins,
                 gamma=DEFAULT_GAMMA, localized=False, verbose=VERBOSE, knowledge=None):
    start = time.time()
    rc, ci_tests = run_multi_phase(normal_df, anomalous_df, gamma, localized, bins, verbose, knowledge)
    end = time.time()

    return {'time': end - start, 'root_cause': rc, 'ci_tests': ci_tests}

def top_k_rc(normal_df, anomalous_df, k, bins,
             gamma=DEFAULT_GAMMA, localized=False, verbose=VERBOSE, knowledge=None):
    result = rca_with_rcd(normal_df, anomalous_df, bins, gamma, localized, verbose, knowledge)
    return {**result, 'root_cause': result['root_cause'][:k]}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PC on the given dataset')

    parser.add_argument('--path', type=str, default=SRC_DIR,
                        help='Path to the experiment data')
    parser.add_argument('--k', type=int, default=K,
                        help='Top-k root causes')
    parser.add_argument('--local', action='store_true',
                        help='Run localized version to only learn the neighborhood of F-node')
    parser.add_argument('--knowledge', type=bool, default=False,
                        help='Run with domain knowledge')
    parser.add_argument('--bench', type=bool, default=False,
                        help='Run bench test')
    parser.add_argument('--all', type=bool, default=False,
                        help='Run run benchtest for all algorithm')
    parser.add_argument('--r', type=int, default=0,
                        help='Set run round times')

    args = parser.parse_args()
    path = args.path
    k = args.k
    local = args.local
    knowledge = args.knowledge
    bench = args.bench
    all = args.all
    r = args.r
    (normal_df, anomalous_df) = u.load_datasets(path + 'normal.csv',
                                                path + 'anomalous.csv')

    # Enable the following line for sock-shop or real outage dataset
    # normal_df, anomalous_df = u.preprocess(normal_df, anomalous_df, 90)
    
    with open(path + 'dependency_list.txt', 'r') as file:
        content = file.read()
    kl = eval(content)

    # result = top_k_rc(normal_df, anomalous_df, k=k, bins=BINS, localized=local, knowledge= kl)
    # result = top_k_rc(normal_df, anomalous_df, k=k, bins=BINS, localized=local)
    if not bench:
        if knowledge:
            result = top_k_rc(normal_df, anomalous_df, k=k, bins=BINS, localized=local, knowledge= kl)
        else:
            result = top_k_rc(normal_df, anomalous_df, k=k, bins=BINS, localized=local)

        print(f"Top {k} took {round(result['time'], 4)} and potential root causes are {result['root_cause']}")
    else:


        # ground_truth = ["X12","X39"]
        # ground_truth = ["carts_cpu", "carts-db_cpu", "carts-db_mem" ,"carts_mem"]
        with open(path + 'root.txt', 'r') as file:
            g_truth_data = file.read()

        # 将读取的字符串按逗号分割，存入列表
        ground_truth = g_truth_data.split(',')


        runs = 10  # Number of runs
        if r: runs = r
        if knowledge==False or all:
            for i in [5]:
                
                times = []  # List to store the execution times

                # Run the function multiple times and store the execution times
                correct_times = 0
                for _ in range(runs):
                    result = top_k_rc(normal_df, anomalous_df, k=i, bins=BINS, localized=local)
                    print(f"Run:{_}, Top {k} took {round(result['time'], 4)} and potential root causes are {result['root_cause']}")
                    times.append(result['time'])
                    for g in ground_truth:
                        if g in result['root_cause']:
                            correct_times += 1
                            break
                # Calculate the average time
                average_time = sum(times) / runs
                correct_precentage = correct_times / runs

                print(f"Average time without knowledge over {runs} runs: {average_time}, correct: {correct_precentage}, k: {i}")


        if knowledge or all:
            for i in [5]:
                times = []  # List to store the execution times
                correct_times = 0
                # Run the function multiple times and store the execution times
                for _ in range(runs):
                    result = top_k_rc(normal_df, anomalous_df, k=i, bins=BINS, localized=local, knowledge=kl)
                    print(f"Run:{_}, Top {k} took {round(result['time'], 4)} and potential root causes are {result['root_cause']}")
                    times.append(result['time'])
                    for g in ground_truth:
                        if g in result['root_cause']:
                            correct_times += 1
                            break
                # Calculate the average time
                average_time = sum(times) / runs
                correct_precentage = correct_times / runs
                print(f"Average time for knowledge over {runs} runs: {average_time}, correct: {correct_precentage}, k: {i}")

