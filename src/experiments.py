import os
import data
import algos
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
        'solution_error': []
        }
        self.t0 = time.perf_counter()
        self.x_true = x_true
        self.tolerance = tolerance

    def record(self, iteration, X, y, lambda_param, w):
        objective_value = algos.objective(X, y, w, lambda_param)
        sparsity_value = algos.sparsity(w)
        residual_value = algos.residual_norm(X, y, w)
        timestamp = time.perf_counter() - self.t0
        solution_error = np.linalg.norm(w - self.x_true) if self.x_true is not None else np.nan

        self.history['iter'].append(iteration)
        self.history['obj'].append(objective_value)
        self.history['sparsity'].append(sparsity_value)
        self.history['residual'].append(residual_value)
        self.history['time'].append(timestamp)
        self.history['solution_error'].append(solution_error)

        print(f"Iter {iteration}: Obj={objective_value:.4f}, Sparsity={sparsity_value:.4f}, Residual={residual_value:.4f}, Time={timestamp:.2f}s")

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
        # Convert to a list of arrays for each field
        data_to_save = {}
        for field in ['iter', 'obj', 'time', 'residual', 'sparsity', 'solution_error']:
            # Each trial's field might have different length -> store as list of arrays
            data_to_save[field] = [np.array(h[field]) for h in histories]
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

    plt.figure()
    for name in algorithms:
        histories = results[name]
        min_len = min(len(h['iter']) for h in histories)
        obj_vals = np.array([h['obj'][:min_len] for h in histories])
        obj_mean = np.mean(obj_vals, axis=0)
        obj_std = np.std(obj_vals, axis=0)
        iters = range(min_len)
        plt.plot(iters, obj_mean, label=name)
        plt.fill_between(iters, obj_mean-obj_std, obj_mean+obj_std, alpha=0.2)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.legend()
    plt.yscale('log')  # often insightful
    plt.show()

    plt.figure()
    for name in algorithms: 
        histories = results[name]
        min_len = min(len(h['time']) for h in histories)
        time_vals = np.array([h['time'][:min_len] for h in histories])
        obj_vals = np.array([h['obj'][:min_len] for h in histories])
        time_mean = np.mean(time_vals, axis=0)
        obj_mean = np.mean(obj_vals, axis=0)
        obj_std = np.std(obj_vals, axis=0)  # 补这一行
        plt.plot(time_mean, obj_mean, label=name)
        plt.fill_between(time_mean, obj_mean-obj_std, obj_mean+obj_std, alpha=0.2)
    plt.xlabel('Time (s)')
    plt.ylabel('Objective Value')
    plt.legend()
    plt.yscale('log')
    plt.show()


def main():
    run_RQ1(num_trials=5)

if __name__ == "__main__":
    main()
