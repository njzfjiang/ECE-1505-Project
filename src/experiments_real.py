from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import metrics
import experiments

def load_and_preprocess_data():
    # Load diabetes dataset
    data = load_diabetes()
    X = data.data
    y = data.target

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def run_real_experiment(num_trials=5, iters=5000, rho=1.0):
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Compute Lipschitz constant and λ_max on training set
    L = np.linalg.eigvalsh(X_train.T @ X_train)[-1]
    lambda_max = np.max(np.abs(X_train.T @ y_train)) / len(y_train)
    lambda_param = 0.1 * lambda_max
    learning_rate = 1.0 / L
    w0 = np.zeros(X_train.shape[1])  # Initial weights

    algorithms = ['ISTA', 'FISTA', 'ADMM']
    all_histories = {name: [] for name in algorithms}
    final_weight_records = {name: [] for name in algorithms}
    eval_metrics = {
        name: {
            'test_mse': [],
            'test_mae': [],
            'test_r2': [],
            'nonzeros': [],
            'train_mse': []
        }
        for name in algorithms
    }

    for trial in range(num_trials):
        rng = np.random.default_rng(trial)
        idx = rng.permutation(len(X_train))
        X_t = X_train[idx]
        y_t = y_train[idx]

        ista_hist, fista_hist, admm_hist, ista_w, fista_w, admm_w = experiments.run_experiment(
            X_t, y_t, x_true=None,
            lambda_param=lambda_param,
            w0=w0,
            learning_rate=learning_rate,
            iters=iters,
            rho=rho
        )

        all_histories['ISTA'].append(ista_hist)
        all_histories['FISTA'].append(fista_hist)
        all_histories['ADMM'].append(admm_hist)

        final_weight_records['ISTA'].append(ista_w)
        final_weight_records['FISTA'].append(fista_w)
        final_weight_records['ADMM'].append(admm_w)

        for name, w in zip(algorithms, [ista_w, fista_w, admm_w]):
            train_pred = X_t @ w
            test_pred = X_test @ w
            eval_metrics[name]['train_mse'].append(mean_squared_error(y_t, train_pred))
            eval_metrics[name]['test_mse'].append(mean_squared_error(y_test, test_pred))
            eval_metrics[name]['test_mae'].append(mean_absolute_error(y_test, test_pred))
            eval_metrics[name]['test_r2'].append(r2_score(y_test, test_pred))
            eval_metrics[name]['nonzeros'].append(metrics.sparsity(w))

    # Aggregate history per algorithm
    def aggregate(histories):
        min_len = min(len(h['iter']) for h in histories)
        iters_mean = np.array(histories[0]['iter'][:min_len])
        obj_vals = np.array([h['obj'][:min_len] for h in histories])
        time_vals = np.array([h['time'][:min_len] for h in histories])

        return {
            'iter': iters_mean,
            'obj_mean': np.mean(obj_vals, axis=0),
            'obj_std': np.std(obj_vals, axis=0),
            'time_mean': np.mean(time_vals, axis=0),
            'time_std': np.std(time_vals, axis=0),
            'raw': histories
        }

    aggregated = {name: aggregate(all_histories[name]) for name in algorithms}

    # Save raw and aggregated data to npz
    for name in algorithms:
        hist_list = all_histories[name]
        save_data = {
            'iter': np.array([np.array(h['iter']) for h in hist_list], dtype=object),
            'obj': np.array([np.array(h['obj']) for h in hist_list], dtype=object),
            'time': np.array([np.array(h['time']) for h in hist_list], dtype=object),
            'residual': np.array([np.array(h['residual']) for h in hist_list], dtype=object),
            'sparsity': np.array([np.array(h['sparsity']) for h in hist_list], dtype=object),
            'solution_error': np.array([np.array(h['solution_error']) for h in hist_list], dtype=object),
            'n_trials': num_trials,
            'lambda_param': lambda_param,
            'rho': rho,
            'iters': iters,
            'obj_mean': aggregated[name]['obj_mean'],
            'obj_std': aggregated[name]['obj_std'],
            'time_mean': aggregated[name]['time_mean'],
            'time_std': aggregated[name]['time_std'],
            'final_weights': np.array(final_weight_records[name], dtype=object),
            'test_mse': np.array(eval_metrics[name]['test_mse']),
            'test_mae': np.array(eval_metrics[name]['test_mae']),
            'test_r2': np.array(eval_metrics[name]['test_r2']),
            'train_mse': np.array(eval_metrics[name]['train_mse']),
            'nonzeros': np.array(eval_metrics[name]['nonzeros']),
            'test_mse_mean': np.mean(eval_metrics[name]['test_mse']),
            'test_mae_mean': np.mean(eval_metrics[name]['test_mae']),
            'test_r2_mean': np.mean(eval_metrics[name]['test_r2']),
            'train_mse_mean': np.mean(eval_metrics[name]['train_mse']),
            'nonzeros_mean': np.mean(eval_metrics[name]['nonzeros']),
        }
        output_fn = f'results/{name.lower()}_real_agg.npz'
        np.savez_compressed(output_fn, **save_data)

    # Print report for metrics
    report_real_metrics(eval_metrics, algorithms)
    return aggregated


def report_real_metrics(eval_metrics, algorithms):
    """Print final metrics for each algorithm over trials."""
    print('\nReal-Data Evaluation Metrics (averaged over trials):')
    print('{:<10} {:>10} {:>10} {:>10} {:>10} {:>10}'.format('Algo', 'Test MSE', 'Test MAE', 'Test R2', 'Nonzeros', 'Train MSE'))
    print('-' * 70)

    report = {}
    for name in algorithms:
        test_mse_mean = np.mean(eval_metrics[name]['test_mse'])
        test_mae_mean = np.mean(eval_metrics[name]['test_mae'])
        test_r2_mean = np.mean(eval_metrics[name]['test_r2'])
        nonzeros_mean = np.mean(eval_metrics[name]['nonzeros'])
        train_mse_mean = np.mean(eval_metrics[name]['train_mse'])

        print('{:<10} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.1f} {:>10.4f}'.format(
            name,
            test_mse_mean,
            test_mae_mean,
            test_r2_mean,
            nonzeros_mean,
            train_mse_mean
        ))

        report[name] = {
            'test_mse': test_mse_mean,
            'test_mae': test_mae_mean,
            'test_r2': test_r2_mean,
            'nonzeros': nonzeros_mean,
            'train_mse': train_mse_mean,
        }

    return report

def plot_real_experiment(aggregated):
    import matplotlib.pyplot as plt

    algorithms = aggregated.keys()

    plt.figure(figsize=(10, 6))
    for name in algorithms:
        plt.plot(aggregated[name]['iter'], aggregated[name]['obj_mean'], label=name, linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Objective')
    plt.title('Average Objective vs Iterations (Real Data)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/real_obj_vs_iter.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    for name in algorithms:
        plt.plot(aggregated[name]['time_mean'], aggregated[name]['obj_mean'], label=name, linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Objective')
    plt.title('Average Objective vs Time (Real Data)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/real_obj_vs_time.png')
    plt.show()

def main():
    aggregated_results = run_real_experiment()
    plot_real_experiment(aggregated_results)

if __name__ == "__main__":
    main()