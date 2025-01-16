from itertools import combinations

import numpy as np
from tqdm.auto import tqdm

from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.cit import chisq, gsq
from causallearn.utils.PCUtils.Helper import append_value
import time


def skeleton_discovery(data, alpha, indep_test, stable=True, background_knowledge=None,
                       labels={}, verbose=False, show_progress=True):
    '''
    Perform skeleton discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    indep_test : the function of the independence test being used
            [fisherz, chisq, gsq, mv_fisherz, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - mv_fisherz: Missing-value Fishers'Z conditional independence test
           - kci: Kernel-based conditional independence test
    stable : run stabilized skeleton discovery if True (default = True)
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.

    Returns
    -------
    cg : a CausalGraph object. Where cg.G.graph[j,i]=0 and cg.G.graph[i,j]=1 indicates  i -> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i -- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    '''

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, labels=labels)
    cg.set_ind_test(indep_test)
    cg.data_hash_key = hash(str(data))
    if indep_test == chisq or indep_test == gsq:
        # if dealing with discrete data, data is numpy.ndarray with n rows m columns,
        # for each column, translate the discrete values to int indexs starting from 0,
        #   e.g. [45, 45, 6, 7, 6, 7] -> [2, 2, 0, 1, 0, 1]
        #        ['apple', 'apple', 'pear', 'peach', 'pear'] -> [0, 0, 2, 1, 2]
        # in old code, its presumed that discrete `data` is already indexed,
        # but here we make sure it's in indexed form, so allow more user input e.g. 'apple' ..
        def _unique(column):
            return np.unique(column, return_inverse=True)[1]

        cg.is_discrete = True
        cg.data = np.apply_along_axis(_unique, 0, data).astype(np.int64)
        cg.cardinalities = np.max(cg.data, axis=0) + 1
    else:
        cg.data = data

    depth = -1
    pbar = tqdm(total=no_of_var) if show_progress else None
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        if show_progress: pbar.reset()
        for x in range(no_of_var):
            if show_progress: pbar.update()
            if show_progress: pbar.set_description(f'Depth={depth}, working on node {x}')
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) < depth - 1:
                continue
            for y in Neigh_x:
                knowledge_ban_edge = False
                sepsets = set()
                if background_knowledge is not None and (
                        background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                        and background_knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])):
                    knowledge_ban_edge = True
                if knowledge_ban_edge:
                    if not stable:
                        edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                        if edge1 is not None:
                            cg.G.remove_edge(edge1)
                        edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                        if edge2 is not None:
                            cg.G.remove_edge(edge2)
                        append_value(cg.sepset, x, y, ())
                        append_value(cg.sepset, y, x, ())
                        break
                    else:
                        edge_removal.append((x, y))  # after all conditioning sets at
                        edge_removal.append((y, x))  # depth l have been considered

                Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                for S in combinations(Neigh_x_noy, depth):
                    p = cg.ci_test(x, y, S)
                    if p > alpha:
                        if verbose: print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                        if not stable:
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                            append_value(cg.sepset, x, y, S)
                            append_value(cg.sepset, y, x, S)
                            break
                        else:
                            edge_removal.append((x, y))  # after all conditioning sets at
                            edge_removal.append((y, x))  # depth l have been considered
                            for s in S:
                                sepsets.add(s)
                    else:
                        append_value(cg.p_values, x, y, p)
                        if verbose: print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
                append_value(cg.sepset, x, y, tuple(sepsets))
                append_value(cg.sepset, y, x, tuple(sepsets))

        if show_progress: pbar.refresh()

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    if show_progress: pbar.close()

    return cg



def local_skeleton_discovery_original(data, local_node, alpha, indep_test, mi=[], labels={}, verbose=False, background_knowledge=None):
    assert type(data) == np.ndarray
    assert local_node <= data.shape[1]
    assert 0 < alpha < 1
    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, labels=labels)
    # print(cg.G.get_nodes())
    cg.set_ind_test(indep_test)
    cg.data_hash_key = hash(str(data))
    if indep_test == chisq or indep_test == gsq:
        def _unique(column):
            return np.unique(column, return_inverse=True)[1]

        cg.is_discrete = True
        cg.data = np.apply_along_axis(_unique, 0, data).astype(np.int64)
        cg.cardinalities = np.max(cg.data, axis=0) + 1
    else:
        cg.data = data

    # orignal code set this to -1, it doesn't check the CI test, only test F-node and y without considering neighbours???
    depth = -1

    x = local_node

    # Remove edges between nodes in MI and F-node
    for i in mi:
        cg.remove_edge(x, i)

    while cg.max_degree() - 1 > depth:
        depth += 1

        local_neigh = np.random.permutation(cg.neighbors(x))
        # local_neigh = cg.neighbors(x)

        if verbose: print("x neigh",local_neigh )

        for y in local_neigh:
            if verbose: print("Y",y)
            Neigh_y = cg.neighbors(y)

            if verbose: print("neighbour of y_1st", Neigh_y)
            Neigh_y = np.delete(Neigh_y, np.where(Neigh_y == x))

            # if background_knowledge is not None:
            #     # Initialize a flag to indicate whether y is in background_knowledge
            #     y_in_background = False
            #     save_node = []
            #     # Iterate over each pair in background_knowledge
            #     for pair in background_knowledge:
            #         if y in pair:
            #             # If y is in the pair, set the flag to True
            #             y_in_background = True

            #             # Determine the other node in the pair
            #             other_node = pair[0] if pair[1] == y else pair[1]
            #             save_node.append(other_node)
            #     for neigh_rm in Neigh_y:
            #         if neigh_rm not in save_node:
            #             # Remove all edges connected to y except the one connected in background_knowledgy
            #             cg.remove_edge(y, neigh_rm)
            #             if verbose: print("remove", neigh_rm)
            #             Neigh_y = np.delete(Neigh_y, np.where(Neigh_y == neigh_rm))
                     

            #     # If y is not in background_knowledge, remove all edges connected to y
            #     if not y_in_background:
            #         for neigh_rm in Neigh_y:
            #             cg.remove_edge(y, neigh_rm)
            #             Neigh_y = np.delete(Neigh_y, np.where(Neigh_y == neigh_rm))


            if background_knowledge is not None:
                # Convert background_knowledge to a set and a dictionary for faster lookup
                background_knowledge_set = set()
                background_knowledge_dict = {}
                for pair in background_knowledge:
                    background_knowledge_set.update(pair)
                    background_knowledge_dict.setdefault(pair[0], []).append(pair[1])
                    background_knowledge_dict.setdefault(pair[1], []).append(pair[0])

                # Initialize a flag to indicate whether y is in background_knowledge
                y_in_background = y in background_knowledge_set

                # Get the nodes to save from the dictionary
                save_node = background_knowledge_dict.get(y, [])

                # Use list comprehension to remove elements from Neigh_y
                Neigh_y = [neigh for neigh in Neigh_y if neigh in save_node]

                # If y is not in background_knowledge, remove all edges connected to y
                if not y_in_background:
                    for neigh_rm in Neigh_y:
                        cg.remove_edge(y, neigh_rm)
                    # Clear Neigh_y since all edges connected to y are removed
                    Neigh_y = []

            if verbose: print("neighbour of y_2nd", Neigh_y)


            Neigh_y_f = []
            if depth > 0:
                if verbose: print("depth",depth)
                for s in Neigh_y:
                    # print("check 1",s, "nei", cg.neighbors(s))
                    if x in cg.neighbors(s):
                        Neigh_y_f.append(s)

                # Neigh_y_f = [s for s in Neigh_y if x in cg.neighbors(s)]

                # Neigh_y_f += mi
            if verbose: print("Neigh_y_f", Neigh_y_f)
            for S in combinations(Neigh_y_f, depth):
                p = cg.ci_test(x, y, S)
                if p > alpha:
                    if verbose: print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                    cg.remove_edge(x, y)
                    append_value(cg.sepset, x, y, S)
                    append_value(cg.sepset, y, x, S)

                    if depth == 0:
                        cg.append_to_mi(y)
                    break
                else:
                    append_value(cg.p_values, x, y, p)
                    if verbose: print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
    return cg

def local_skeleton_discovery_1st_optimize(data, local_node, alpha, indep_test, mi=[], labels={}, verbose=False, background_knowledge=None):
    assert type(data) == np.ndarray
    assert local_node < data.shape[1]
    assert 0 < alpha < 1
    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, labels=labels)
    cg.set_ind_test(indep_test)
    cg.data_hash_key = hash(str(data))

    if indep_test == chisq or indep_test == gsq:
        def _unique(column):
            return np.unique(column, return_inverse=True)[1]

        cg.is_discrete = True
        cg.data = np.apply_along_axis(_unique, 0, data).astype(np.int64)
        cg.cardinalities = np.max(cg.data, axis=0) + 1
    else:
        cg.data = data

    depth = -1
    x = local_node

    # Remove edges between nodes in MI and F-node
    for i in mi:
        cg.remove_edge(x, i)

    # Convert background knowledge to a set for faster lookup
    allowed_edges = set()
    if background_knowledge is not None:
        for pair in background_knowledge:
            node_a, node_b = pair
            allowed_edges.add((node_a, node_b))
            allowed_edges.add((node_b, node_a))

    while cg.max_degree() - 1 > depth:
        depth += 1

        local_neigh = np.random.permutation(cg.neighbors(x))
        if verbose: print("x neigh", local_neigh)

        for y in local_neigh:
            if verbose: print("Y", y)
            Neigh_y = cg.neighbors(y)
            if verbose: print("neighbour of y_1st", Neigh_y)
            Neigh_y = np.delete(Neigh_y, np.where(Neigh_y == x))

            # 保留背景知识中允许的边
            Neigh_y = [neigh for neigh in Neigh_y if (y, neigh) in allowed_edges or (neigh, y) in allowed_edges]

            if verbose: print("neighbour of y_2nd", Neigh_y)

            Neigh_y_f = []
            if depth > 0:
                if verbose: print("depth", depth)
                for s in Neigh_y:
                    if x in cg.neighbors(s):
                        Neigh_y_f.append(s)

            if verbose: print("Neigh_y_f", Neigh_y_f)
            for S in combinations(Neigh_y_f, depth):
                p = cg.ci_test(x, y, S)
                if p > alpha:
                    if verbose: print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                    cg.remove_edge(x, y)
                    append_value(cg.sepset, x, y, S)
                    append_value(cg.sepset, y, x, S)

                    if depth == 0:
                        cg.append_to_mi(y)
                    break
                else:
                    append_value(cg.p_values, x, y, p)
                    if verbose: print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
    return cg

def local_skeleton_discovery(data, local_node, alpha, indep_test, mi=[], labels={}, verbose=False, background_knowledge=None):
    assert type(data) == np.ndarray
    assert local_node <= data.shape[1]
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, labels=labels)
    cg.set_ind_test(indep_test)
    cg.data_hash_key = hash(str(data))
    if indep_test == chisq or indep_test == gsq:
        def _unique(column):
            return np.unique(column, return_inverse=True)[1]

        cg.is_discrete = True
        cg.data = np.apply_along_axis(_unique, 0, data).astype(np.int64)
        cg.cardinalities = np.max(cg.data, axis=0) + 1
    else:
        cg.data = data

    depth = -1
    x = local_node
    # Remove edges between nodes in MI and F-node
    for i in mi:
        cg.remove_edge(x, i)

    # Convert background knowledge to a set for faster lookup
    allowed_edges = set()
    if background_knowledge is not None:
        for pair in background_knowledge:
            node_a, node_b = pair
            allowed_edges.add((node_a, node_b))
            allowed_edges.add((node_b, node_a))
        # all_edges = cg.find_undirected()
        # for edge in all_edges:
        #     if (edge[0], edge[1]) not in allowed_edges and (edge[1], edge[0]) not in allowed_edges and (edge[0]!=x and edge[1]!=x):
        #         cg.remove_edge(edge[0], edge[1])
        #         cg.remove_edge(edge[1], edge[0])
                




    # print("list",undirected)

    # Cache for conditional independence test results
    ci_test_cache = {}

    def cached_ci_test(x, y, S):
        key = (x, y, tuple(sorted(S)))  # Generate a unique key for the test
        if key not in ci_test_cache:
            ci_test_cache[key] = cg.ci_test(x, y, S)
        return ci_test_cache[key]

    while cg.max_degree() - 1 > depth:
        depth += 1

        local_neigh = np.random.permutation(cg.neighbors(x))
        # local_neigh = cg.neighbors(x)
        for y in local_neigh:
            Neigh_y = cg.neighbors(y)
            Neigh_y = np.delete(Neigh_y, np.where(Neigh_y == x))

            # if background_knowledge is not None:
            #     # print("allowed_edges: ", allowed_edges)
            #     # 保留背景知识中允许的边
            #     Neigh_y = [neigh for neigh in Neigh_y if (y, neigh) in allowed_edges or (neigh, y) in allowed_edges]


            Neigh_y_f = []
            if depth > 0:
                Neigh_y_f = [s for s in Neigh_y if x in cg.neighbors(s)]
                # Neigh_y_f += mi

            for S in combinations(Neigh_y_f, depth):
                if background_knowledge is not None:
                    if (x, y) in allowed_edges or (y, x) in allowed_edges:
                        p = cached_ci_test(x, y, S)  # Use the cached CI test
                        # p=0
                        if verbose: print(f'{cg.labels[x]} and {cg.labels[y]} are known to be dependent by background knowledge.')
                    else:
                        if x != local_node and y != local_node:
                            p = alpha + 0.1
                        else: 
                            p = cached_ci_test(x, y, S)  # Use the cached CI test
                else:
                    p = cg.ci_test(x, y, S)
                if p > alpha:
                    if verbose: print(f'{cg.labels[x]} ind {cg.labels[y]} | {[cg.labels[s] for s in S]} with p-value {p}')
                    cg.remove_edge(x, y)
                    append_value(cg.sepset, x, y, S)
                    append_value(cg.sepset, y, x, S)

                    if depth == 0:
                        cg.append_to_mi(y)
                    break
                else:
                    append_value(cg.p_values, x, y, p)
                    if verbose: print(f'{cg.labels[x]} dep {cg.labels[y]} | {[cg.labels[s] for s in S]} with p-value {p}')

    return cg


def local_skeleton_discovery_modify(data, local_node, alpha, indep_test, mi=[], labels={}, verbose=False, background_knowledge=None):
    start = time.time()
    assert type(data) == np.ndarray
    assert local_node < data.shape[1]
    assert 0 < alpha < 1
    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, labels=labels)
    cg.set_ind_test(indep_test)
    cg.data_hash_key = hash(str(data))

    if indep_test in [chisq, gsq]:
        def _unique(column):
            return np.unique(column, return_inverse=True)[1]

        cg.is_discrete = True
        cg.data = np.apply_along_axis(_unique, 0, data).astype(np.int64)
        cg.cardinalities = np.max(cg.data, axis=0) + 1
    else:
        cg.data = data

    depth = -1
    x = local_node

    # Remove edges between nodes in MI and F-node
    for i in mi:
        cg.remove_edge(x, i)

    # Convert background knowledge to a set for faster lookup
    allowed_edges = set()
    if background_knowledge is not None:
        for pair in background_knowledge:
            node_a, node_b = pair
            allowed_edges.add((node_a, node_b))
            allowed_edges.add((node_b, node_a))

    # print(time.time()-start)
    start = time.time()
    # Cache for conditional independence test results
    ci_test_cache = {}

    def cached_ci_test(x, y, S):
        key = (x, y, tuple(sorted(S)))  # Generate a unique key for the test
        if key not in ci_test_cache:
            ci_test_cache[key] = cg.ci_test(x, y, S)
        return ci_test_cache[key]

    while cg.max_degree() - 1 > depth:
        depth += 1

        local_neigh = np.random.permutation(cg.neighbors(x))
        # if verbose: print("x neigh", local_neigh)

        for y in local_neigh:
            # if verbose: print("Y", y)
            Neigh_y = cg.neighbors(y)
            # if verbose: print("neighbour of y_1st", Neigh_y)
            Neigh_y = np.delete(Neigh_y, np.where(Neigh_y == x))

            if background_knowledge is not None:
                # 保留背景知识中允许的边
                Neigh_y = [neigh for neigh in Neigh_y if (y, neigh) in allowed_edges or (neigh, y) in allowed_edges]

            # print("neighbour of y_2nd", Neigh_y)

            Neigh_y_f = []
            if depth > 0:
                if verbose: print("depth", depth)
                for s in Neigh_y:
                    if x in cg.neighbors(s):
                        Neigh_y_f.append(s)

            # if verbose: print("Neigh_y_f", Neigh_y_f)
            for S in combinations(Neigh_y_f, depth):
                p = cached_ci_test(x, y, S)  # Use the cached CI test
                if p > alpha:
                    if verbose: print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                    cg.remove_edge(x, y)
                    append_value(cg.sepset, x, y, S)
                    append_value(cg.sepset, y, x, S)

                    if depth == 0:
                        cg.append_to_mi(y)
                    break
                else:
                    append_value(cg.p_values, x, y, p)
                    if verbose: print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
    return cg

def local_skeleton_discovery_3rd_optimize(data, local_node, alpha, indep_test, mi=[], labels={}, verbose=False, background_knowledge=None): # not best
    assert type(data) == np.ndarray
    assert local_node <= data.shape[1]
    assert 0 < alpha < 1
    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, labels=labels)
    cg.set_ind_test(indep_test)
    cg.data_hash_key = hash(str(data))
    
    ci_test_cache = {}  # Cache for storing CI test results
    
    if indep_test == chisq or indep_test == gsq:
        def _unique(column):
            return np.unique(column, return_inverse=True)[1]

        cg.is_discrete = True
        cg.data = np.apply_along_axis(_unique, 0, data).astype(np.int64)
        cg.cardinalities = np.max(cg.data, axis=0) + 1
    else:
        cg.data = data

    # Preprocess background knowledge for quick lookup
    background_knowledge_set = set()
    background_knowledge_dict = {}
    if background_knowledge is not None:
        for pair in background_knowledge:
            background_knowledge_set.update(pair)
            background_knowledge_dict.setdefault(pair[0], []).append(pair[1])
            background_knowledge_dict.setdefault(pair[1], []).append(pair[0])

    depth = -1
    x = local_node

    # Remove edges between nodes in MI and F-node
    for i in mi:
        cg.remove_edge(x, i)

    while cg.max_degree() - 1 > depth:
        depth += 1
        local_neigh = np.random.permutation(cg.neighbors(x))
        if verbose: print("x neigh", local_neigh)

        for y in local_neigh:
            if verbose: print("Y", y)
            Neigh_y = cg.neighbors(y)
            if verbose: print("neighbour of y_1st", Neigh_y)
            Neigh_y = np.delete(Neigh_y, np.where(Neigh_y == x))

            # Apply background knowledge
            if y in background_knowledge_set:
                save_node = background_knowledge_dict.get(y, [])
                Neigh_y = [neigh for neigh in Neigh_y if neigh in save_node]
            else:
                Neigh_y = []

            if verbose: print("neighbour of y_2nd", Neigh_y)

            Neigh_y_f = []
            if depth > 0:
                if verbose: print("depth", depth)
                for s in Neigh_y:
                    if x in cg.neighbors(s):
                        Neigh_y_f.append(s)

            if verbose: print("Neigh_y_f", Neigh_y_f)
            for S in combinations(Neigh_y_f, depth):
                p = cached_ci_test(cg, x, y, S, ci_test_cache)
                if p > alpha:
                    if verbose: print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                    cg.remove_edge(x, y)
                    append_value(cg.sepset, x, y, S)
                    append_value(cg.sepset, y, x, S)
                    if depth == 0:
                        cg.append_to_mi(y)
                    break
                else:
                    append_value(cg.p_values, x, y, p)
                    if verbose: print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
    return cg

def cached_ci_test(cg, x, y, S, ci_test_cache):
    key = (x, y, tuple(sorted(S)))  # Generate a unique key for the test
    if key not in ci_test_cache:
        ci_test_cache[key] = cg.ci_test(x, y, S)
    return ci_test_cache[key]