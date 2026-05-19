"""
Params optimized by HPO, summary:
name: "greedy_ratio", importance: 0.6694, optimized_value: 0.6388624291680867
name: "rev_prob", importance: 0.1670, optimized_value: 0.33542965610294795
name: "perm_prob", importance: 0.1635, optimized_value: 0.2609731527001532
"""

import numpy as np
import itertools
import os
import random
import math
import time
from openevolve.hpo import tunable

# EVOLVE-BLOCK-START

def get_neighbors_c7_k(n, k):
    num_nodes = n**k
    powers = (n**np.arange(k)).astype(np.int32)
    nodes = (np.arange(num_nodes, dtype=np.int32)[:, None] // powers) % n
    shifts = np.array(list(itertools.product([-1, 0, 1], repeat=k)), dtype=np.int8)
    shifts = shifts[np.any(shifts != 0, axis=1)]
    adj = np.dot((nodes[:, None, :] + shifts) % n, powers)
    return adj, nodes, powers

def solve_greedy(n, k, adj, nodes, powers, rev_indices, time_limit=10.0):
    start_time, best_is, num_nodes = time.time(), [], n**k
    available, base_order = np.ones(num_nodes, bool), np.arange(num_nodes, dtype=np.int32)
    adj_list = [row for row in adj]
    
    # Tuned defaults from HPO summary of top performing Program 1 (Score: 0.9977)
    rev_prob = tunable("rev_prob", range=(0.0, 1.0), default=0.33542965610294795)
    perm_prob = tunable("perm_prob", range=(0.0, 0.5), default=0.2609731527001532)

    while time.time() - start_time < time_limit:
        available.fill(True)
        if random.random() < perm_prob:
            p = np.random.permutation(k)
            s = np.random.randint(0, n, size=k)
            current_base_order = np.dot((nodes[:, p] + s) % n, powers).astype(np.int32)
        else:
            current_base_order = base_order

        step = random.randint(1, num_nodes - 1)
        while step % n == 0: step = random.randint(1, num_nodes - 1)
        order = (current_base_order * step + random.randint(0, num_nodes - 1)) % num_nodes
        
        if random.random() < rev_prob: order = rev_indices[order]
            
        curr = []
        for v in order.tolist():
            if available[v]:
                curr.append(v)
                available[adj_list[v]] = False
        if len(curr) > len(best_is):
            best_is = curr
    return best_is

def local_search(n, k, adj, best_is, time_limit):
    if not best_is: return []
    num_nodes = n**k
    curr_is = list(best_is)
    is_set = set(curr_is)
    tightness = np.zeros(num_nodes, dtype=np.int16)
    
    # Vectorized tightness initialization from high-performing Program 1
    if curr_is:
        all_neighbors = adj[curr_is].flatten()
        np.add.at(tightness, all_neighbors, 1)
    
    adj_sets = [set(row) for row in adj]
    
    best_is_ever = list(best_is) # Initialize with the best set from greedy
    start_time = time.time()
    while time.time() - start_time < time_limit:
        if not curr_is: break
        idx = random.randrange(len(curr_is))
        u = curr_is[idx]
        
        # Remove u
        last = curr_is[-1]
        curr_is[idx] = last
        curr_is.pop()
        is_set.remove(u)
        
        u_neighbors = adj[u]
        tightness[u_neighbors] -= 1
        
        # Find candidates (neighbors of u with tightness 0)
        cands_mask = (tightness[u_neighbors] == 0)
        candidates = u_neighbors[cands_mask]
        
        replacement_is = []
        if len(candidates) > 0:
            cand_set = set(candidates)
            cand_degrees = []
            for v_cand in candidates:
                # Calculate degree within the candidate-induced subgraph
                degree = sum(1 for neighbor in adj[v_cand] if neighbor in cand_set)
                cand_degrees.append((v_cand, degree))

            # Greedily build the IS from low-degree nodes in the induced subgraph.
            cand_degrees.sort(key=lambda x: (x[1], random.random()))
            
            cand_is_set = set() # This set tracks the IS *within* the candidates
            for v, _ in cand_degrees:
                # Use fast set disjoint check for independence
                if adj_sets[v].isdisjoint(cand_is_set):
                    replacement_is.append(v)
                    cand_is_set.add(v)
        
        # If we found a 1-for-k swap (k>=0, where k is len(replacement_is)), apply it.
        if replacement_is:
            for v_add in replacement_is:
                curr_is.append(v_add)
                is_set.add(v_add) # Keep is_set consistent
                tightness[adj[v_add]] += 1
        else:
            # No viable replacement found among candidates, so revert the removal of u.
            curr_is.append(u)
            is_set.add(u) # Keep is_set consistent
            tightness[u_neighbors] += 1
        
        # Always track the best IS found so far
        if len(curr_is) > len(best_is_ever):
            best_is_ever = list(curr_is)
                
    return best_is_ever

def generate_independent_set(n: int, k: int) -> list:
    """Generates an independent set for the strong product graph C_n^k."""
    adj, nodes, powers = get_neighbors_c7_k(n, k)
    num_nodes = n**k
    
    # Tuned default from HPO summary of top performing Program 1 (Score: 0.9977)
    greedy_ratio = tunable("greedy_ratio", range=(0.1, 0.9), default=0.6388624291680867)
    t_total = 19.0
    t_greedy = t_total * greedy_ratio
    t_local = t_total - t_greedy
    
    indices = np.arange(num_nodes, dtype=np.int32)
    rev_indices = np.zeros_like(indices)
    temp_indices = indices.copy()
    for _ in range(k):
        rev_indices = rev_indices * n + (temp_indices % n)
        temp_indices //= n
        
    best_indices = solve_greedy(n, k, adj, nodes, powers, rev_indices, time_limit=t_greedy)
    final_indices = local_search(n, k, adj, best_indices, time_limit=t_local)
    
    return nodes[final_indices].tolist()

# EVOLVE-BLOCK-END
