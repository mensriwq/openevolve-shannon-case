import sys
import os
import time
import importlib.util
import random
import numpy as np

class MockHPO:
    @staticmethod
    def tunable(name, *args, **kwargs):
        if 'default' in kwargs:
            return kwargs['default']
        
        if args:
            if isinstance(args[0], tuple):
                if len(args) >= 2:
                    return args[1] # default
            else:
                if len(args) >= 3:
                    return args[2] # default
                    
        return None

mock_mod = type(sys)('openevolve.hpo')
mock_mod.tunable = MockHPO.tunable
sys.modules['openevolve.hpo'] = mock_mod

def check_independent_set(nodes, n=7):
    size = len(nodes)
    if size == 0:
        return False
    
    k = len(nodes[0])
    nodes_array = np.array(nodes)
    
    for i in range(size):
        u = nodes_array[i]
        others = nodes_array[i+1:]
        if len(others) == 0:
            continue
            
        diff = np.abs(others - u)
        dist = np.minimum(diff, n - diff)
        is_neighbor = np.all(dist <= 1, axis=1)
        
        if np.any(is_neighbor):
            return False
            
    return True

def save_results_to_file(script_path, elapsed, result_is, is_valid, score):
    base_name = os.path.splitext(os.path.basename(script_path))[0]
    output_file = f"{base_name}.txt"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Verification Report ===\n")
            f.write(f"Script: {os.path.basename(script_path)}\n")
            f.write(f"Status: {'SUCCESS' if is_valid else 'FAILED'}\n")
            f.write(f"Score (Size): {score}\n")
            f.write(f"Execution Time: {elapsed:.2f}s\n")
            f.write(f"Number of Nodes: {len(result_is)}\n")
            f.write("-" * 30 + "\n")
            f.write("Full Independent Set Nodes:\n")
            for node in result_is:
                f.write(f"{node}\n")
                
        print(f"[*] Full results and node list saved to: {output_file}")
    except Exception as e:
        print(f"[!] Error saving to file: {e}")

def run_and_verify(script_path, n=7, k=5, seed=42, greedy_iters=6330, local_iters=1000000):
    if not os.path.exists(script_path):
        print(f"[!] File not found: {script_path}")
        return

    spec = importlib.util.spec_from_file_location("target_script", script_path)
    module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"[!] Error executing module: {e}")
        import traceback
        traceback.print_exc()
        return

    if not hasattr(module, 'generate_independent_set'):
        print("[!] `generate_independent_set(n, k)` not found")
        return

    try:
        random.seed(seed)
        np.random.seed(seed)

        print(f"[*] Running {os.path.basename(script_path)} with greedy_iters={greedy_iters}, local_iters={local_iters}...")
        start_search_time = time.time()
        result_is, actual_it = module.generate_independent_set(n, k, greedy_iters=greedy_iters, local_iters=local_iters)
        elapsed = time.time() - start_search_time

        is_valid = check_independent_set(result_is, n)
        score = len(result_is) if is_valid else 0
        
        print(f"[*] Finished in {elapsed:.2f}s")
        print(f"[*] Result length: {len(result_is)}")
        print(f"[*] Actual Local It : {actual_it}")
        print("-" * 30)
        print(f"State: {'success' if is_valid else 'failed'}")
        print(f"Score: {score}")
        print("-" * 30)
        
        save_results_to_file(script_path, elapsed, result_is, is_valid, score)
        return score, actual_it, result_is
        
    except Exception as e:
        print(f"[!] Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verifier.py <program.py> [spectrum/search/binary]")
        sys.exit(1)
    
    script = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "default"
    
    if mode == "deep_search":
        print("[*] Deep Search mode for finding unique spectra (Score >= 367)...")
        output_log = "all_sota_solutions.txt"
        with open(output_log, "w") as f:
            f.write("Greedy_It\tLocal_It\tDist-2\n")
            f.write("-" * 40 + "\n")
            
        found_spectra = set()
        local_limit = 100000
        
        # Searching range 1601 to 2000 for uniqueness
        for g in range(1601, 2001):
            score, it, nodes = run_and_verify(script, seed=42, greedy_iters=g, local_iters=local_limit)
            if score >= 367:
                # Calculate dist-2
                distances = []
                nodes_array = np.array(nodes)
                for i in range(len(nodes_array)):
                    u = nodes_array[i]
                    others = nodes_array[i+1:]
                    if len(others) == 0: continue
                    diff = np.abs(others - u)
                    dist_dims = np.minimum(diff, 7 - diff)
                    d_maxs = np.max(dist_dims, axis=1)
                    distances.extend(d_maxs.tolist())
                d2 = sum(1 for d in distances if d == 2)
                
                with open(output_log, "a") as f:
                    f.write(f"{g}\t\t{it}\t\t{d2}\n")
                    
                if d2 not in found_spectra:
                    found_spectra.add(d2)
                    print(f"[NEW SPECTRUM] Greedy: {g}, Dist-2: {d2}")
                
                if len(found_spectra) >= 15: # Stop if we found enough variety
                    if g > 1800:
                        break
                        
        print(f"\n[!] Finished. Found {len(found_spectra)} unique Dist-2 spectra.")
        
    elif mode == "spectrum":
        print("[*] Starting spectrum analysis for multiple configurations...")
        configs = [1474, 1506, 1510, 1518, 1525, 1544, 1577, 1599, 1600]
        local_limit = 200000
        
        output_dir = "eval_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"\n{'Greedy_It':<10} | {'Score':<5} | {'Actual_It':<10} | {'Saved To'}")
        print("-" * 75)
        
        for g in configs:
            score, it, nodes = run_and_verify(script, seed=42, greedy_iters=g, local_iters=local_limit)
            
            # Save nodes for cross-verification
            save_path = os.path.join(output_dir, f"nodes_g{g}.npy")
            np.save(save_path, np.array(nodes))
            
            print(f"{g:<10} | {score:<5} | {it:<10} | {save_path}")
            
    elif mode == "binary":
        print("[*] Starting binary search for emergence point (Score >= 360)...")
        low = 100
        high = 3000
        emergence_point = high
        local_limit = 100000
        
        while low <= high:
            mid = (low + high) // 2
            score, it, _ = run_and_verify(script, seed=42, greedy_iters=mid, local_iters=local_limit)
            print(f"--- Binary Check: Greedy {mid} -> Score {score} ---")
            if score >= 360:
                emergence_point = mid
                high = mid - 1
            else:
                low = mid + 1
        
        print(f"\n[!] Emergence Point Found: greedy_iters = {emergence_point}")
    elif mode == "search":
        print("[*] Starting search for optimal greedy_iters (fixed seed=42)...")
        local_limit = 100000
        test_range = [1432, 1550] # 1432 is the Emergence Point
        results = []
        
        for g in test_range:
            score, it, _ = run_and_verify(script, seed=42, greedy_iters=g, local_iters=local_limit)
            print(f"# Greedy It {g}: Score {score}, Actual Local It {it}")
            results.append((g, score, it))
        
        print("\n=== Search Results ===")
        print("Greedy_Iters\tScore\tLocal_It")
        for g, s, it in sorted(results, key=lambda x: x[2]):
            print(f"{g}\t\t{s}\t\t{it}")
    else:
        run_and_verify(script, seed=42, greedy_iters=1525, local_iters=32182) # 1525 is the fastest lucky SOTA point

# Greedy It 1474: Score 367, Actual Local It 85306
# Greedy It 1506: Score 367, Actual Local It 85520
# Greedy It 1510: Score 367, Actual Local It 41761
# Greedy It 1518: Score 367, Actual Local It 83496
# Greedy It 1525: Score 367, Actual Local It 32182
# Greedy It 1544: Score 367, Actual Local It 75710
# Greedy It 1577: Score 367, Actual Local It 85888
# Greedy It 1599: Score 367, Actual Local It 35715
# Greedy It 1600: Score 367, Actual Local It 96454
