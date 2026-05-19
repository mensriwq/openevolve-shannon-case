import os
import re
import ast
import tempfile
import optuna
import psutil
import yaml
import threading
from typing import Dict, Any, Tuple, Optional, Callable, Union, List, Iterator
from dataclasses import dataclass, field
from contextlib import contextmanager

from openevolve.evaluation_result import EvaluationResult

import subprocess

# Global context for runtime HPO sampling and parameter injection
_hpo_ctx = threading.local()

# Monkey-patch subprocess.Popen to automatically inject thread-local HPO parameters
# This enables true multithreaded parallel HPO search for C++ without os.environ race conditions
_original_popen_init = subprocess.Popen.__init__

def _hpo_popen_init(self, args, **kwargs):
    # Check if there's a thread-local hpo env
    hpo_env = getattr(_hpo_ctx, "params_env", None)
    if hpo_env:
        # Get the existing env or process environ
        env = kwargs.get("env")
        if env is None:
            env = os.environ.copy()
        else:
            env = env.copy()
        
        # Inject HPO env vars
        for k, v in hpo_env.items():
            env[f"HPO_{k}"] = str(v)
            
        kwargs["env"] = env
        
    _original_popen_init(self, args, **kwargs)

# Apply patch
if subprocess.Popen.__init__ is not _hpo_popen_init:
    subprocess.Popen.__init__ = _hpo_popen_init


def get_current_trial() -> Optional[optuna.trial.Trial]:
    return getattr(_hpo_ctx, "trial", None)

def set_current_trial(trial: Optional[optuna.trial.Trial]):
    _hpo_ctx.trial = trial

def get_hpo_params() -> Optional[Dict[str, Any]]:
    return getattr(_hpo_ctx, "params", None)

def set_hpo_params(params: Optional[Dict[str, Any]]):
    _hpo_ctx.params = params

@contextmanager
def hpo_env_context(params: Dict[str, Any]) -> Iterator[None]:
    """Context manager to securely set thread-local HPO parameters for subprocess monkey-patch."""
    old_params_env = getattr(_hpo_ctx, "params_env", None)
    try:
        _hpo_ctx.params_env = params
        yield
    finally:
        _hpo_ctx.params_env = old_params_env

def hpo_print(message: str, hpo_log_path: Optional[str] = None):
    """Print to console and append to a dedicated HPO log file if provided."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    
    # 1. Always print to console for real-time visibility (even in subprocess)
    print(message)
    
    # 2. Append to shared HPO log file if path is known
    if hpo_log_path:
        try:
            with open(hpo_log_path, "a") as f:
                f.write(full_message + "\n")
        except Exception:
            pass

@dataclass
class HPOConfig:
    """Internal HPO configuration with default values"""
    enabled: bool = False
    trigger_metric: str = "combined_score"
    objective_metric: str = "combined_score"
    threshold_score: float = 0.8
    num_trials: int = 50
    stage_trials: List[float] = field(default_factory=lambda: [1.0, 3.0, 10.0])
    time_limit_per_run: float = 20.0
    improvement_threshold: float = 0.5
    n_jobs: Any = "auto"
    random_seed: int = 42
    report_importance: bool = True

def tunable(name: str, min: Any, max: Any, default: Any, log: bool = False, **kwargs) -> Any:
    """
    Standard stub for tunable parameters. 
    In runtime, it prioritizes sources in this order:
    1. Optuna trial (during HPO search)
    2. Injected params from context (during final evaluation or re-evaluation)
    3. Default value (fallback)
    
    Args:
        name: A unique name for this parameter. MANDATORY.
        min: The minimum value of the range.
        max: The maximum value of the range.
        default: The default value to use when not tuning.
        log: Whether to use log-scale sampling.
    """
    if name is None or not isinstance(name, str):
        raise ValueError("The 'name' parameter for 'tunable()' is mandatory and must be a unique string.")
        
    # 1. Optuna Trial (Highest priority during search)
    trial = get_current_trial()
    if trial is not None:
        # Determine sampling method
        if isinstance(min, int) and isinstance(max, int):
            return trial.suggest_int(name, min, max)
        else:
            # Auto-detect log scale if not specified
            if not log and min > 0 and max / min >= 100:
                log = True
            return trial.suggest_float(name, min, max, log=log)
            
    # 2. Injected Parameters (Priority during evaluation)
    params = get_hpo_params()
    if params is not None and name in params:
        return params[name]

    # 3. Default value (Fallback)
    return default

# Type for core evaluation function
# Modified to accept optional kwargs for hpo_mode and hpo_context
EvaluationFunction = Callable[..., Union[Dict[str, Any], EvaluationResult]]

def _get_comment_style(language: str) -> Tuple[str, str]:
    """Return comment start/end markers for different languages."""
    if language in ["cpp", "c++", "java", "javascript", "rust"]:
        return "/*", "*/"
    return '"""', '"""'

def _detect_language(file_path: Optional[str], code: Optional[str] = None) -> str:
    """Detect programming language based on file extension or heuristics."""
    if file_path:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".cpp", ".cc", ".cxx", ".h", ".hpp"]: return "cpp"
        if ext in [".py"]: return "python"
        if ext in [".js", ".ts"]: return "javascript"
        if ext in [".java"]: return "java"
        if ext in [".rs"]: return "rust"
    
    if code:
        if "#include" in code or "using namespace" in code: return "cpp"
        if "import " in code or "def " in code: return "python"
        
    return "python"

def update_program_params(code: str, params: Dict[str, Any], importances: Optional[Dict[str, float]] = None, language: str = "python") -> str:
    """
    Update parameters in code by finding 'tunable(...)' calls via AST (Python only)
    and modifying the 'default' argument using direct string replacement on precise coordinates.
    Also adds a summary of optimized parameters at the top of the file using correct language comments.
    """
    sig = "Params optimized by HPO, summary:"
    comment_start, comment_end = _get_comment_style(language)
    
    try:
        # 1. Remove existing HPO params block if present (support both python and cpp comment styles)
        hpo_block_pattern = rf'(?s)({re.escape(comment_start)}|""")\s*{re.escape(sig)}.*?name\s*:.*?({re.escape(comment_end)}|""")\s*'
        code = re.sub(hpo_block_pattern, "", code, flags=re.DOTALL)

        # 2. Parse AST to find tunable calls (Python Only)
        if language == "python":
            try:
                tree = ast.parse(code)
            except SyntaxError:
                print("HPO Error: Failed to parse code for update_program_params")
                return code

            replacements =[]

            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'tunable':
                    # Extract parameter name
                    p_name = None
                    if len(node.args) >= 1 and isinstance(node.args[0], ast.Constant):
                        p_name = node.args[0].value
                    
                    for kw in node.keywords:
                        if kw.arg == 'name' and isinstance(kw.value, ast.Constant):
                            p_name = kw.value.value
                    
                    if not p_name:
                        p_name = f"tunable_{node.lineno}_{node.col_offset}"
                    
                    if p_name in params:
                        new_val = params[p_name]
                        
                        # Find 'default' argument node
                        default_node = None
                        
                        for kw in node.keywords:
                            if kw.arg == 'default':
                                default_node = kw.value
                                break
                        
                        if not default_node and len(node.args) >= 4:
                            default_node = node.args[3]

                        if default_node:
                            new_val_repr = repr(new_val)
                            replacements.append({
                                "start_lineno": default_node.lineno,
                                "start_col": default_node.col_offset,
                                "end_lineno": default_node.end_lineno,
                                "end_col": default_node.end_col_offset,
                                "new_text": new_val_repr
                            })

            # 3. Apply Python AST replacements
            replacements.sort(key=lambda x: (x["start_lineno"], x["start_col"]), reverse=True)
            
            lines = code.splitlines(keepends=True)
            
            for rep in replacements:
                line_idx = rep["start_lineno"] - 1
                end_line_idx = rep["end_lineno"] - 1
                
                if line_idx == end_line_idx:
                    line = lines[line_idx]
                    pre = line[:rep["start_col"]]
                    post = line[rep["end_col"]:]
                    lines[line_idx] = pre + rep["new_text"] + post
                else:
                    lines[line_idx] = lines[line_idx][:rep["start_col"]] + rep["new_text"]
                    for i in range(line_idx + 1, end_line_idx):
                        lines[i] = "" 
                    lines[end_line_idx] = lines[end_line_idx][rep["end_col"]:]

            updated_code = "".join(lines)
        else:
            updated_code = code
            if language == "cpp":
                for p_name, new_val in params.items():
                    pattern = rf'(get_tunable(?:_log)?\s*\(\s*"{re.escape(p_name)}"\s*,\s*[^,]+,\s*[^,]+,\s*)([^)]+)(\))'
                    if isinstance(new_val, float):
                        val_str = f"{new_val}f" if "f" in updated_code else str(new_val)
                    else:
                        val_str = str(new_val)
                    updated_code = re.sub(pattern, rf'\g<1>{val_str}\g<3>', updated_code)

        # 4. Add HPO params summary block at the top
        if importances:
            important_items = sorted([(k, v) for k, v in importances.items() if v > 0.1],
                key=lambda x: x[1],
                reverse=True
            )
            
            has_unimportant = any(v <= 0.1 for v in importances.values())
            
            if important_items or has_unimportant:
                summary_lines = [comment_start, sig]
                for name, imp in important_items:
                    val = params.get(name, "N/A")
                    summary_lines.append(f'name: "{name}", importance: {imp:.4f}, optimized_value: {val}')
                
                if has_unimportant:
                    summary_lines.append("Remaining parameters have importance below 0.1, consider removing them and introducing other parameter structures for exploration.")
                
                summary_lines.append(comment_end)
                
                updated_code = "\n".join(summary_lines) + "\n\n" + updated_code

        return updated_code
    except Exception as e:
        print(f"HPO Update Error: {e}")
        return code


def get_optimal_n_jobs(config_n_jobs: Any) -> int:
    """Determine optimal number of jobs based on system load"""
    if isinstance(config_n_jobs, int) and config_n_jobs != -1:
        return config_n_jobs
    try:
        cpu_count = psutil.cpu_count(logical=False) or os.cpu_count() or 1

        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        cpu_usage = psutil.cpu_percent(interval=0.1)
        if mem.percent > 85 or swap.percent > 40 or cpu_usage > 95:
            return 2
        if cpu_usage > 80.0:
            return max(1, int(cpu_count * 0.25))
        if mem.percent < 60:
            return max(1, int(cpu_count * 0.75))
        else:
            return max(1, int(cpu_count * 0.5))

    except Exception:
        return 1


class HPOObjective:
    """Objective function that enables runtime sampling via global context"""
    def __init__(self, original_code: str, evaluate_fn: EvaluationFunction, program_dir: str, objective_metric: str, program_path: Optional[str] = None, probed_params: Optional[Dict[str, Any]] = None):
        self.original_code = original_code
        self.evaluate_fn = evaluate_fn
        self.program_dir = program_dir
        self.objective_metric = objective_metric
        self.program_path = program_path
        self.language = _detect_language(program_path, original_code)
        self.probed_params = probed_params or {}

    def __call__(self, trial: optuna.trial.Trial) -> float:
        code_to_run = self.original_code
        
        ext_map = {
            "python": ".py",
            "cpp": ".cpp", 
            "c++": ".cpp",
            "rust": ".rs",
            "java": ".java",
            "javascript": ".js"
        }
        suffix = ext_map.get(self.language, ".py")
        if self.program_path:
            _, ext = os.path.splitext(self.program_path)
            if ext: suffix = ext

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, dir=self.program_dir, delete=False) as tf:
                tf.write(code_to_run)
                temp_path = tf.name
            
            set_current_trial(trial)
            try:
                params_env = {}
                if self.language == "cpp":
                    # Use probed parameters (either from static regex or dynamic probe)
                    for p_name, p_info in self.probed_params.items():
                        val = trial.suggest_float(p_name, p_info["min"], p_info["max"], log=p_info["log"])
                        params_env[p_name] = val
                    
                    # Fallback/Safety: Try static regex for parameters not captured by dynamic probing
                    # This ensures we still capture parameters even if probing didn't execute that code path
                    pattern = r'get_tunable(_log)?\s*\(\s*"([^"]+)"\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\)'
                    for match in re.finditer(pattern, code_to_run):
                        is_log = bool(match.group(1))
                        p_name = match.group(2)
                        
                        # If already captured by dynamic probing or provided via suggested_params, skip static parsing
                        if p_name in params_env:
                            continue
                            
                        p_min_str = match.group(3).replace('f', '').replace('F', '').strip()
                        p_max_str = match.group(4).replace('f', '').replace('F', '').strip()
                        try:
                            p_min = float(p_min_str)
                            p_max = float(p_max_str)
                            
                            # Robustness: Auto-swap if low > high to prevent Optuna failure
                            actual_min = min(p_min, p_max)
                            actual_max = max(p_min, p_max)
                            
                            val = trial.suggest_float(p_name, actual_min, actual_max, log=is_log and actual_min > 0)
                            params_env[p_name] = val
                        except ValueError:
                            # Only warn if we haven't successfully probed this parameter
                            # If probing is enabled and this param isn't in params_env, it might be an expression/variable
                            if p_name not in params_env and p_name not in self.probed_params:
                                print(f"Warning: Failed to parse static bounds for '{p_name}'. "
                                      f"Ensure bounds are numeric constants or use dynamic probing.")
                        except Exception as e:
                            if p_name not in params_env and p_name not in self.probed_params:
                                print(f"Warning: Failed to parse bounds for {p_name}: {e}")

                with hpo_env_context(params_env):
                    # Always use fast mode for HPO search. 
                    # This enforces project evaluators to support the 'hpo_mode' argument.
                    result = self.evaluate_fn(temp_path, hpo_mode=True)
            finally:
                set_current_trial(None)

            metrics = result.metrics if isinstance(result, EvaluationResult) else result
            return float(metrics.get(self.objective_metric, 0.0))
        except Exception:
            return -10.0
        finally:
            if temp_path and os.path.exists(temp_path):
                try: os.remove(temp_path)
                except: pass


def run_hpo(
    program_path: str,
    original_code: str,
    current_score: float,
    hpo_config: HPOConfig,
    evaluate_fn: EvaluationFunction,
    suggested_params: Optional[Dict[str, Any]] = None,
    hpo_log_path: Optional[str] = None,
    probed_params: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], float, Optional[Dict[str, Any]]]:
    """
    Run HPO using runtime dynamic sampling.
    Returns: (optimized_code, best_score, best_params)
    """
    language = _detect_language(program_path, original_code)
    
    if language == "python" and "tunable" not in original_code:
        return None, current_score, None
    elif language == "cpp" and "get_tunable" not in original_code:
        return None, current_score, None
        
    n_jobs = get_optimal_n_jobs(hpo_config.n_jobs)
    
    hpo_print(f"HPO: Starting optimization with {n_jobs} jobs and InMemory storage...", hpo_log_path)
    
    try:
        # Use InMemoryStorage for better performance in transient HPO tasks
        storage = optuna.storages.InMemoryStorage()
        
        program_dir = os.path.dirname(os.path.abspath(program_path))
        objective = HPOObjective(original_code, evaluate_fn, program_dir, hpo_config.objective_metric, program_path=program_path, probed_params=probed_params)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=hpo_config.random_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler, storage=storage)
        
        # --- Path 3: Inject suggested params as a seed trial if available ---
        if suggested_params:
            hpo_print("HPO: Enqueuing suggested params as initial seed trial.", hpo_log_path)
            study.enqueue_trial(suggested_params)
            
        best_so_far = current_score
        improved_any_stage = False
        
        for stage_idx, multiplier in enumerate(hpo_config.stage_trials):
            stage_trials = int(hpo_config.num_trials * multiplier)
            hpo_print(f"HPO Stage {stage_idx + 1}: Running {stage_trials} trials...", hpo_log_path)
            
            study.optimize(objective, n_trials=stage_trials, timeout=hpo_config.time_limit_per_run * stage_trials / n_jobs, n_jobs=n_jobs)
            best_value = study.best_value

            if best_value > best_so_far + hpo_config.improvement_threshold:
                hpo_print(f"HPO Stage {stage_idx + 1}: Improvement found ({best_so_far:.4f} -> {best_value:.4f}).", hpo_log_path)
                best_so_far = best_value
                improved_any_stage = True
            else:
                hpo_print(f"HPO Stage {stage_idx + 1}: No improvement in this stage. Stopping HPO stages.", hpo_log_path)
                break
        
        if improved_any_stage:
            best_params = study.best_params
            importances = None
            if hpo_config.report_importance:
                importances = optuna.importance.get_param_importances(study)

            language = _detect_language(program_path, original_code)
            hpo_print(f"HPO: Final improvement ({current_score:.4f} -> {best_so_far:.4f}). Applying new default values.", hpo_log_path)
            final_code = update_program_params(original_code, best_params, importances, language=language)
            
            return final_code, best_so_far, best_params

    except Exception as e:
        hpo_print(f"HPO failed: {e}", hpo_log_path)
            
    return None, current_score, None


def evaluate_with_hpo(
    program_path: str,
    core_evaluate_fn: EvaluationFunction,
    objective_metric: str = "combined_score",
    hpo_context: Optional[Dict[str, Any]] = None
) -> EvaluationResult:
    """
    Entry point for project evaluators.
    Uses 3-path strategy for robustness and reproducibility:
    
    Path 1 (LLM Reference): Parameter summary at top of code (via code rewrite)
    Path 2 (LLM Reference): In-place parameter replacement (via code rewrite)
        Note: Path 1 & 2 are best-effort for LLM learning and may not be perfect due to code complexity.
    Path 3 (Evaluation Source of Truth): Artifacts 'hpo_best_params' injected at runtime
        This ensures reproducibility and accuracy regardless of code rewrite success.
        
    Args:
        program_path: Path to program file
        core_evaluate_fn: Function to evaluate the program
        objective_metric: Metric to optimize
        hpo_context: Dictionary containing 'config' (dict) and 'suggested_params' (dict).
                     This is the SSOT provided by the OpenEvolve framework.
    """
    # Logging setup
    import logging
    logger = logging.getLogger(__name__)
    
    hpo_config_dict = {
        "enabled": False,
        "trigger_metric": objective_metric,
        "objective_metric": objective_metric,
        "threshold_score": 0.8,
        "num_trials": 50,
        "stage_trials": [1.0, 3.0, 10.0],
        "time_limit_per_run": 20.0,
        "improvement_threshold": 0.001,
        "n_jobs": "auto",
        "random_seed": 42
    }
    
    # Load config from context (SSOT)
    if hpo_context and "config" in hpo_context and hpo_context["config"]:
        logger.debug("HPO: Using HPO config from context.")
        hpo_config_dict.update(hpo_context["config"])

    hpo_config = HPOConfig(**hpo_config_dict)

    # --- Path 3 Logic: Check if we already have suggested parameters (Source of Truth) ---
    # We use these params as a Seed for HPO later.
    suggested_params = None
    if hpo_context and "suggested_params" in hpo_context:
        suggested_params = hpo_context["suggested_params"]

    # --- Standard Flow: Initial Evaluation -> HPO Search -> Final Evaluation ---
    
    # Pre-evaluation Probe Setup
    probe_file_path = None
    probed_params = {}
    if hpo_config.enabled and _detect_language(program_path) == "cpp":
        try:
            fd, probe_file_path = tempfile.mkstemp(suffix=".txt", prefix="hpo_probe_")
            os.close(fd)
        except Exception:
            probe_file_path = None

    # Run initial evaluation (this will now also act as our HPO probe for C++)
    with hpo_env_context({"OPENEVOLVE_HPO_PROBE_FILE": probe_file_path} if probe_file_path else {}):
        initial_res = core_evaluate_fn(program_path)
    
    final_res = initial_res if isinstance(initial_res, EvaluationResult) else EvaluationResult(metrics=initial_res)

    # Post-evaluation Probe Extraction
    if probe_file_path and os.path.exists(probe_file_path):
        try:
            with open(probe_file_path, "r") as f:
                for line in f:
                    if line.startswith("[HPO_PARAM]"):
                        parts = line.replace("[HPO_PARAM]", "").strip().split(":")
                        if len(parts) >= 5:
                            p_name = parts[0].strip()
                            try:
                                probed_params[p_name] = {
                                    "min": float(parts[1]),
                                    "max": float(parts[2]),
                                    "default": float(parts[3]),
                                    "log": parts[4].strip() == "log"
                                }
                            except ValueError:
                                continue
        except Exception as e:
            logger.warning(f"Failed to read HPO probe file: {e}")
        finally:
            try:
                os.remove(probe_file_path)
            except Exception:
                pass

    if hpo_config.enabled:
        trigger_metric = hpo_config.trigger_metric
        current_score = final_res.metrics.get(trigger_metric, 0.0)
        
        if current_score >= hpo_config.threshold_score:
            with open(program_path, 'r') as f:
                original_code = f.read()
            
            # Run HPO Search
            # We pass suggested_params to HPO as a seed to ensure reproducibility 
            # while allowing for further optimization if conditions changed.
            optimized_code, best_score, best_params = run_hpo(
                program_path=program_path,
                original_code=original_code,
                current_score=current_score,
                hpo_config=hpo_config,
                evaluate_fn=core_evaluate_fn,
                suggested_params=suggested_params,
                hpo_log_path=hpo_context.get("hpo_log_path"),
                probed_params=probed_params
            )
            
            if best_params:
                # --- Path 3: Inject best params for accurate final evaluation ---
                set_hpo_params(best_params)
                try:
                    # Final evaluation with Source of Truth parameters
                    # Inject for subprocesses (C++) as well
                    with hpo_env_context(best_params):
                        final_res_raw = core_evaluate_fn(program_path)
                    final_res = final_res_raw if isinstance(final_res_raw, EvaluationResult) else EvaluationResult(metrics=final_res_raw)
                    
                    final_score = final_res.metrics.get(hpo_config.objective_metric, 0.0)
                    
                    # Check for discrepancies
                    if abs(final_score - best_score) > hpo_config.improvement_threshold:
                        logger.error(f"HPO Error: Significant discrepancy between HPO best score ({best_score:.4f}) and final eval score ({final_score:.4f}). A potential issue with reproducibility")
                    
                    final_res.metrics["hpo_tuned"] = True
                    
                    # Path 1 & 2: Store modified code for LLM reference
                    final_res.artifacts["optimized_code"] = optimized_code
                    
                    # Path 3: Store Source of Truth parameters for reproducibility
                    final_res.artifacts["hpo_best_params"] = best_params
                    final_res.artifacts["hpo_best_score"] = best_score
                    
                finally:
                    set_hpo_params(None)
    
    return final_res
