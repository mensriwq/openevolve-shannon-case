import sys
import os
import time
import numpy as np
import itertools
import yaml
import statistics
import math
import re
import ast
import random
import json
import uuid
import subprocess
import tempfile
import numba as nb
from openevolve.evaluation_result import EvaluationResult

try:
    import torch
except ImportError:
    torch = None

# Configuration defaults
N_DEFAULT = 7
K_DEFAULT = 5
SEED_DEFAULT = 42
TARGET_BOUND_DEFAULT = 367.0

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def create_default_metrics():
    return {
        "num_points": 0,
        "n_eff": 0,
        "coordinate_std": 0.0,
        "neighbor_count_std": 0.0,
        "combined_score": 0.0,
        "eval_time": 0.0,
        "error": 0
    }

def _calculate_normalized_score(penalized_score, n, k, target_bound=TARGET_BOUND_DEFAULT):
    trivial_bound = (n // 2)**k
    if penalized_score <= 0: return 0.0
    if penalized_score <= trivial_bound:
        return (penalized_score / trivial_bound) * 0.1 if trivial_bound > 0 else 0.0
    elif penalized_score <= target_bound:
        return 0.1 + ((penalized_score - trivial_bound) / (target_bound - trivial_bound)) * 0.9
    else:
        return 1.0 + (penalized_score - target_bound) / target_bound

@nb.njit(fastmath=True, cache=True)
def _check_conflicts_and_get_neighbors(points, n, k):
    """
    Simultaneously check for conflicts and count neighbors with distance == 2.
    - Points are adjacent (conflict) if they are close in all dimensions (diff <= 1 or diff == n-1).
    - Neighbor count is based on max dimension distance == 2.
    """
    m = points.shape[0]
    n_minus_1 = n - 1
    counts = np.zeros(m, dtype=np.int32)
    
    for i in range(m):
        for j in range(i + 1, m):
            is_adjacent = True
            max_dist = 0
            for d in range(k):
                diff = abs(points[i, d] - points[j, d])
                # Conflict condition: adjacent in all dimensions
                if diff > 1 and diff < n_minus_1:
                    is_adjacent = False
                
                # Metric for distance == 2
                dist = diff if diff < n - diff else n - diff
                if dist > max_dist:
                    max_dist = dist
            
            if is_adjacent:
                return True, counts # Conflict found
            
            if max_dist == 2:
                counts[i] += 1
                counts[j] += 1
                
    return False, counts

def calculate_coordinate_std(points, n):
    if len(points) == 0: return 0.0
    # points is already a numpy array (int32)
    counts = np.bincount(points.ravel(), minlength=n)
    return float(np.std(counts))

def evaluate_single_seed(program_path, n, k, seed, target_bound=TARGET_BOUND_DEFAULT, hpo_mode=False):
    metrics = create_default_metrics()
    try:
        points = []
        eval_time = 0.0
        
        # --- C++ Path ---
        if program_path.endswith(".cpp"):
            from openevolve.utils.cpp_utils import compile_and_run_cpp
            wrapper_path = os.path.join(THIS_DIR, "main_wrapper.cpp")
            success, output, run_time = compile_and_run_cpp(
                program_path=[program_path, wrapper_path],
                hpo_context=None,
                compiler="g++",
                optimization="-O3 -march=native",
                timeout=60,
                run_args=[n, k]
            )
            eval_time = run_time
            if not success:
                metrics["error"] = output; return metrics
            # Parse output
            points = []
            for line in output.strip().split('\n'):
                if not line.strip(): continue
                try:
                    pt = [int(x) for x in line.strip().split()]
                    if len(pt) == k: points.append(pt)
                except ValueError: pass
        # --- Python Path ---
        elif program_path.endswith(".py"):
            try:
                with open(program_path, 'r', encoding='utf-8') as f:
                    original_code = f.read()
                
                exec_namespace = {
                    "n": n,
                    "k": k,
                    "__builtins__": __builtins__
                }
                injected_code = f"n = {n}\nk = {k}\n" + original_code
                
                st = time.process_time()
                exec(injected_code, exec_namespace)
                
                if "generate_independent_set" not in exec_namespace:
                    metrics["error"] = "Function 'generate_independent_set' not found"
                    return metrics
                    
                points = exec_namespace["generate_independent_set"](n, k)
                eval_time = time.process_time() - st
            except Exception as e:
                metrics["error"] = f"Runtime error: {str(e)}"
                return metrics
            
            if points is None:
                metrics["error"] = "Program returned None"; return metrics
        else:
            metrics["error"] = "Unsupported file type"; return metrics

        # --- Common Post-processing ---
        points = np.asarray(points, dtype=np.int32)
        num_points = len(points)
        if num_points == 0:
            metrics["eval_time"] = float(eval_time); return metrics

        # 1. Illegal check (Cheap)
        theta_cn = (n * math.cos(math.pi/n)) / (1 + math.cos(math.pi/n))
        lovasz_bound = theta_cn ** k
        if num_points > lovasz_bound + 1:
             metrics["error"] = f"Illegal program: points exceeding Lovász bound"; metrics["combined_score"] = -1.0; return metrics

        # 2. Conflict & Feature calculation
        if hpo_mode:
            n_eff = num_points
            coordinate_std = neighbor_count_std = 0.0
        else:
            has_conflict, neighbor_counts = _check_conflicts_and_get_neighbors(points, n, k)
            if has_conflict:
                n_eff = 0
                coordinate_std = 0.0
                neighbor_count_std = 0.0
            else:
                n_eff = num_points
                coordinate_std = calculate_coordinate_std(points, n)
                neighbor_count_std = float(np.std(neighbor_counts)) if len(neighbor_counts) > 0 else 0.0

        penalized_score = n_eff - min(0.99, eval_time / 60.0)
        combined_score = _calculate_normalized_score(penalized_score, n, k, target_bound=target_bound)

        metrics.update({
            "num_points": int(num_points), "n_eff": int(n_eff),
            "coordinate_std": coordinate_std,
            "neighbor_count_std": neighbor_count_std,
            "combined_score": float(combined_score), "eval_time": float(eval_time),
        })
        return metrics
    except Exception as e:
        metrics["error"] = str(e); metrics["combined_score"] = -1.0; return metrics

def _get_config():
    n, k, seed, target_bound = N_DEFAULT, K_DEFAULT, SEED_DEFAULT, TARGET_BOUND_DEFAULT
    config_path = os.path.join(os.path.dirname(THIS_DIR), "config.yaml")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
                if "shannon" in config_data:
                    n, k = config_data["shannon"].get("n", n), config_data["shannon"].get("k", k)
                    target_bound = config_data["shannon"].get("target_bound", target_bound)
                seed = config_data.get("random_seed", seed)
        except: pass
    return n, k, seed, target_bound

from openevolve.hpo import evaluate_with_hpo
def evaluate(program_path: str, hpo_context: dict = None):
    n, k, seed, target_bound = _get_config()
    def core_fn(path, hpo_mode=False):
        return evaluate_single_seed(path, n, k, seed, target_bound=target_bound, hpo_mode=hpo_mode)
    return evaluate_with_hpo(program_path, core_fn, hpo_context=hpo_context)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = evaluate(sys.argv[1])
        print(json.dumps(result.to_dict() if hasattr(result, "to_dict") else vars(result)))
