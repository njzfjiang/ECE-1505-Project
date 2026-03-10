import os
import data
import algos
import metrics
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

os.makedirs('results', exist_ok=True)


class Recorder:
    """
    A simple class to record the history of optimization.
    """
    def __init__(self, x_true=None, tolerance=1e-6):
        self.history = {
        'iter': [],
        'obj': [],
        'residual': [],
        'sparsity': [],
        'time': [],
        'solution_error': [],
        # ADMM-specific diagnostics will be stored here if provided
        'primal_residual': [],
        'dual_residual': []
        }
        self.t0 = time.perf_counter()
        self.x_true = x_true
        self.tolerance = tolerance

    def record(self, iteration, X, y, lambda_param, w, **kwargs):
        objective_value = metrics.objective(X, y, w, lambda_param)
        sparsity_value = metrics.sparsity(w)
        residual_value = metrics.residual_norm(X, y, w)
        timestamp = time.perf_counter() - self.t0
        solution_error = np.linalg.norm(w - self.x_true) if self.x_true is not None else np.nan

        self.history['iter'].append(iteration)
        self.history['obj'].append(objective_value)
        self.history['sparsity'].append(sparsity_value)
        self.history['residual'].append(residual_value)
        self.history['time'].append(timestamp)
        self.history['solution_error'].append(solution_error)

        # record any extra information passed by the callback
        self.history['primal_residual'].append(kwargs.get('primal_residual', np.nan))
        self.history['dual_residual'].append(kwargs.get('dual_residual', np.nan))

        # print(f"Iter {iteration}: Obj={objective_value:.4f}, Sparsity={sparsity_value:.4f}, Residual={residual_value:.4f}, Time={timestamp:.2f}s")

        # early stopping condition (example: if objective value changes very little) 
        if iteration > 0 and abs(self.history['obj'][-1] - self.history['obj'][-2])/(self.history['obj'][-2]+1e-8) < self.tolerance:
            return True
        return False

def run_experiment(X, y, x_true, lambda_param, w0, iters, rho, learning_rate):
    # Create a recorder to track the optimization history
    ista_recorder = Recorder(x_true=x_true, tolerance=1e-6)
    # Run ISTA for LASSO regression
    ista_w_estimated = algos.ista_lasso(X, y, lambda_param, w0, iters, learning_rate, callback=ista_recorder.record)
    
    fista_recorder = Recorder(x_true=x_true, tolerance=1e-6)
    # Run FISTA for LASSO regression
    fista_w_estimated = algos.fista_lasso(X, y, lambda_param, w0, iters, learning_rate, callback=fista_recorder.record)

    admm_recorder = Recorder(x_true=x_true, tolerance=1e-6)
    # Run ADMM for LASSO regression
    admm_w_estimated = algos.admm_lasso(X, y, lambda_param, w0, iters, rho, callback=admm_recorder.record)


    return ista_recorder.history, fista_recorder.history, admm_recorder.history, ista_w_estimated, fista_w_estimated, admm_w_estimated


def run_RQ1(num_trials=5):
    n = 100
    d = 200
    cond_number = 10
    sparsity = 0.1
    noise_level = 0.1
    struct = 'iid'
    lambda_param = 0.1
    w0 = np.zeros(d)
    iters = 5000
    rho = 1.0

    
    algorithms = {
    'ISTA': algos.ista_lasso,
    'FISTA': algos.fista_lasso,
    'ADMM': algos.admm_lasso
    }

    results = {name: [] for name in algorithms}

    for trial in range(num_trials): 
        np.random.seed(trial) # Run multiple trials to average results
        # Generate synthetic data
        X, y, x_true = data.generate_data(n, d, cond_number, sparsity, noise_level, struct)

        L = np.linalg.eigvalsh(X.T @ X)[-1]
        step0 = 1.0 / L

        ista_history, fista_history, admm_history, ista_w, fista_w, admm_w = run_experiment(
            X, y, x_true, lambda_param, w0, iters, rho, step0)
        
        for name, history in zip(['ISTA', 'FISTA', 'ADMM'], [ista_history, fista_history, admm_history]):
            results[name].append(history)
    
    for name in algorithms:
        histories = results[name]
        min_len = min(len(h['iter']) for h in histories)
        # Collect arrays
        obj_vals = np.array([h['obj'][:min_len] for h in histories])
        time_vals = np.array([h['time'][:min_len] for h in histories])
        # Compute mean and std
        obj_mean = np.mean(obj_vals, axis=0)
        obj_std = np.std(obj_vals, axis=0)
        time_mean = np.mean(time_vals, axis=0)

        # Save raw histories for all trials
        histories = results[name]  # list of dicts, each dict has 'iter','obj','time',...
        # Convert to a list of arrays for each field; use object dtype because lengths vary
        data_to_save = {}
        for field in ['iter', 'obj', 'time', 'residual', 'sparsity', 'solution_error']:
            data_to_save[field] = np.array([np.array(h[field]) for h in histories],
                                           dtype=object)
        # Also save metadata: number of trials, algorithm name, parameters
        data_to_save['n_trials'] = len(histories)
        data_to_save['algorithm'] = name
        data_to_save['n'] = n
        data_to_save['d'] = d
        data_to_save['cond_number'] = cond_number
        data_to_save['sparsity'] = sparsity
        data_to_save['noise_level'] = noise_level
        data_to_save['lambda_param'] = lambda_param
        data_to_save['rho'] = rho
        
        # Save to NPZ file
        np.savez_compressed(f'results/{name}_raw.npz', **data_to_save)



def run_RQ2(num_trials=5):
    #How does the conditioning of A (well-conditioned vs. highly correlated features) affect the relative performance of these methods?
    condition_numbers = [5, 10, 50, 100, 500]
    structures = ['iid', 'toeplitz']
    n = 100
    d = 200
    sparsity = 0.1
    noise_level = 0.1
    lambda_param = 0.1
    w0 = np.zeros(d)
    iters = 5000
    rho = 1.0

    algorithms = {
    'ISTA': algos.ista_lasso,
    'FISTA': algos.fista_lasso,
    'ADMM': algos.admm_lasso
    }

    # loop over each configuration and run multiple trials
    for cond in condition_numbers:
        for struct in structures:
            trial_results = {name: [] for name in algorithms}
            for trial in range(num_trials):
                np.random.seed(trial)
                X, y, x_true = data.generate_data(n, d, cond, sparsity, noise_level, struct)
                L = np.linalg.eigvalsh(X.T @ X)[-1]
                step0 = 1.0 / L
                ista_history, fista_history, admm_history, *_ = run_experiment(
                    X, y, x_true, lambda_param, w0, iters, rho, step0)
                for name, history in zip(['ISTA', 'FISTA', 'ADMM'],
                                         [ista_history, fista_history, admm_history]):
                    trial_results[name].append(history)

            # aggregate results for each algorithm
            for name in algorithms:
                histories = trial_results[name]
                min_len = min(len(h['iter']) for h in histories)
                obj_vals = np.array([h['obj'][:min_len] for h in histories])
                time_vals = np.array([h['time'][:min_len] for h in histories])
                iter_counts = np.array([h['iter'][-1] if h['iter'] else np.nan for h in histories])
                time_counts = np.array([h['time'][-1] if h['time'] else np.nan for h in histories])

                data_to_save = {
                    'n_trials': num_trials,
                    'algorithm': name,
                    'n': n,
                    'd': d,
                    'cond_number': cond,
                    'structure': struct,
                    'sparsity': sparsity,
                    'noise_level': noise_level,
                    'lambda_param': lambda_param,
                    'rho': rho,
                    'obj_mean': np.mean(obj_vals, axis=0),
                    'obj_std': np.std(obj_vals, axis=0),
                    'time_mean': np.mean(time_vals, axis=0),
                    'time_std': np.std(time_vals, axis=0),
                    'iters_mean': np.mean(iter_counts),
                    'iters_std': np.std(iter_counts),
                    'time_to_tol_mean': np.mean(time_counts),
                    'time_to_tol_std': np.std(time_counts)
                }
                filename = f'results/{struct}_cond{cond}_{name.lower()}_agg.npz'
                np.savez_compressed(filename, **data_to_save)
                print(f"Saved aggregated results to {filename}")

def run_RQ3(num_trials=5):
    #How does the choice of hyperparameters (e.g., step size for ISTA/FISTA, penalty parameter for ADMM) influence convergence speed and solution quality?
    # grid search over hyperparameters, aggregating results across multiple seeds
    cond = 50
    struct = 'iid'
    n = 100
    d = 200
    sparsity = 0.1
    noise_level = 0.1
    lambda_param = 0.1
    w0 = np.zeros(d)
    iters = 5000
    alphas = [0.1, 0.5, 0.9, 1.0, 1.1]  # step sizes for ISTA/FISTA
    rhos = [0.01, 0.1, 1, 10, 100]  # penalty parameters for ADMM

    algorithms = {
    'ISTA': algos.ista_lasso,
    'FISTA': algos.fista_lasso,
    'ADMM': algos.admm_lasso
    }

    # ISTA/FISTA sweep
    for alpha in alphas:
        # collect trials separately for each method
        trial_data = {'ISTA': [], 'FISTA': []}
        for trial in range(num_trials):
            np.random.seed(trial)
            X, y, x_true = data.generate_data(n, d, cond, sparsity, noise_level, struct)
            L = np.linalg.eigvalsh(X.T @ X)[-1]
            step0 = alpha / L
            ista_rec = Recorder(x_true=x_true, tolerance=1e-6)
            algos.ista_lasso(X, y, lambda_param, w0, iters, step0, callback=ista_rec.record)
            fista_rec = Recorder(x_true=x_true, tolerance=1e-6)
            algos.fista_lasso(X, y, lambda_param, w0, iters, step0, callback=fista_rec.record)
            trial_data['ISTA'].append(ista_rec.history)
            trial_data['FISTA'].append(fista_rec.history)

        # aggregate and save for each method
        for name in ['ISTA', 'FISTA']:
            histories = trial_data[name]
            min_len = min(len(h['iter']) for h in histories)
            iters_arr = np.array([h['iter'][-1] if h['iter'] else np.nan for h in histories])
            times_arr = np.array([h['time'][-1] if h['time'] else np.nan for h in histories])
            obj_vals = np.array([h['obj'][:min_len] for h in histories])

            data_to_save = {
                'algorithm': name,
                'alpha': alpha,
                'n_trials': num_trials,
                'n': n,
                'd': d,
                'cond': cond,
                'sparsity': sparsity,
                'noise_level': noise_level,
                'lambda_param': lambda_param,
                'iters_mean': np.nanmean(iters_arr),
                'iters_std': np.nanstd(iters_arr),
                'time_mean': np.nanmean(times_arr),
                'time_std': np.nanstd(times_arr),
                'obj_curve_mean': np.mean(obj_vals, axis=0),
                'obj_curve_std': np.std(obj_vals, axis=0)
            }
            filename = f'results/{name.lower()}_alpha{alpha}_agg.npz'
            np.savez_compressed(filename, **data_to_save)
            print(f"Saved aggregated {name} (alpha={alpha}) to {filename}")

    # ADMM sweep
    for rho in rhos:
        trial_hist = []
        for trial in range(num_trials):
            np.random.seed(trial)
            X, y, x_true = data.generate_data(n, d, cond, sparsity, noise_level, struct)
            admm_rec = Recorder(x_true=x_true, tolerance=1e-6)
            algos.admm_lasso(X, y, lambda_param, w0, iters, rho, callback=admm_rec.record)
            trial_hist.append(admm_rec.history)

        min_len = min(len(h['iter']) for h in trial_hist)
        iters_arr = np.array([h['iter'][-1] if h['iter'] else np.nan for h in trial_hist])
        times_arr = np.array([h['time'][-1] if h['time'] else np.nan for h in trial_hist])
        primal_mat = np.array([h['primal_residual'][:min_len] for h in trial_hist])
        dual_mat = np.array([h['dual_residual'][:min_len] for h in trial_hist])

        data_to_save = {
            'algorithm': 'ADMM',
            'rho': rho,
            'n_trials': num_trials,
            'n': n,
            'd': d,
            'cond': cond,
            'sparsity': sparsity,
            'noise_level': noise_level,
            'lambda_param': lambda_param,
            'iters_mean': np.nanmean(iters_arr),
            'iters_std': np.nanstd(iters_arr),
            'time_mean': np.nanmean(times_arr),
            'time_std': np.nanstd(times_arr),
            'primal_mean': np.mean(primal_mat, axis=0),
            'primal_std': np.std(primal_mat, axis=0),
            'dual_mean': np.mean(dual_mat, axis=0),
            'dual_std': np.std(dual_mat, axis=0)
        }
        filename = f'results/admm_rho{rho}_agg.npz'
        np.savez_compressed(filename, **data_to_save)
        print(f"Saved aggregated ADMM (rho={rho}) to {filename}")

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Run all experiments (RQ1, RQ2, RQ3)')
    parser.add_argument('--rq1', action='store_true', help='Run RQ1 experiments')
    parser.add_argument('--rq2', action='store_true', help='Run RQ2 experiments')
    parser.add_argument('--rq3', action='store_true', help='Run RQ3 experiments')
    args = parser.parse_args(argv)

    if args.all:
        run_RQ1(num_trials=5)
        run_RQ2(num_trials=5)
        run_RQ3(num_trials=5)
    elif args.rq1:
        run_RQ1(num_trials=5)
    elif args.rq2:
        run_RQ2(num_trials=5)
    elif args.rq3:
        run_RQ3(num_trials=5)
    else:       
        print("Please specify which experiment to run: --rq1, --rq2, --rq3, or --all")

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
