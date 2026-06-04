import os
import json
import time
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler, RandomSampler
from sklearn.model_selection import TimeSeriesSplit
from log import log

optuna.logging.set_verbosity(optuna.logging.WARNING)


class MeowTuner:

    def __init__(self, output_dir='tuning_output'):
        self.output_dir = output_dir
        self.trial_logs = []
        self.top5_models = []
        os.makedirs(output_dir, exist_ok=True)

    def _suggest_params(self, trial):
        max_depth = trial.suggest_int('max_depth', 4, 6)
        num_leaves_max = min(2 ** max_depth, 63)
        num_leaves = trial.suggest_int('num_leaves', 15, num_leaves_max)
        learning_rate = trial.suggest_float('learning_rate', 0.005, 0.08, log=True)
        n_estimators = trial.suggest_int('n_estimators', 500, 2000, step=100)
        subsample = trial.suggest_float('subsample', 0.5, 0.8)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 0.8)
        min_child_samples = trial.suggest_int('min_child_samples', 50, 300, step=50)
        reg_alpha = trial.suggest_float('reg_alpha', 0.01, 2.0, log=True)
        reg_lambda = trial.suggest_float('reg_lambda', 0.01, 2.0, log=True)
        early_stopping_rounds = max(50, int(0.1 / learning_rate * 50))
        return {
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'subsample_freq': 1,
            'min_child_samples': min_child_samples,
            'min_child_weight': 1e-3,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'early_stopping_rounds': early_stopping_rounds,
        }

    @staticmethod
    def _eval_pearson(y_true, y_pred):
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        if np.var(y_true) < 1e-8 or np.var(y_pred) < 1e-8:
            return 0.0
        return np.corrcoef(y_true, y_pred)[0, 1]

    @staticmethod
    def _eval_r2(y_true, y_pred):
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        sse = np.sum((y_pred - y_true) ** 2)
        var_y = np.var(y_true)
        n = len(y_true)
        if var_y < 1e-8 or n == 0:
            return 0.0
        return 1 - sse / var_y / n

    @staticmethod
    def _eval_mse(y_true, y_pred):
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        return np.sum((y_pred - y_true) ** 2) / len(y_true)

    def _train_and_evaluate_with_model(self, params, X_train, y_train, X_val, y_val,
                                        early_stopping_rounds=50):
        from mdl import MeowModel
        dynamic_esr = params.get('early_stopping_rounds', early_stopping_rounds)
        params_copy = {k: v for k, v in params.items() if k != 'early_stopping_rounds'}
        model = MeowModel(cacheDir=None)
        model.update_params(params_copy)
        model.update_params({'early_stopping_rounds': dynamic_esr})
        xdf_train = pd.DataFrame(X_train, columns=[f'f{i}' for i in range(X_train.shape[1])])
        ydf_train = pd.DataFrame(y_train, columns=['target'])
        xdf_val = pd.DataFrame(X_val, columns=[f'f{i}' for i in range(X_val.shape[1])])
        ydf_val = pd.DataFrame(y_val, columns=['target'])
        model.fit_with_validation(xdf_train, ydf_train, xdf_val, ydf_val)
        y_val_pred = model.predict(xdf_val)
        y_train_pred = model.predict(xdf_train)
        pearson = self._eval_pearson(y_val, y_val_pred)
        r2 = self._eval_r2(y_val, y_val_pred)
        mse = self._eval_mse(y_val, y_val_pred)
        train_pearson = self._eval_pearson(y_train, y_train_pred)
        best_iteration = model.model.best_iteration if model.model else 0
        return pearson, r2, mse, best_iteration, train_pearson

    def _objective(self, trial, X, y, n_splits=5, early_stopping_rounds=50):
        params = self._suggest_params(trial)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_pearsons, fold_r2s, fold_mses = [], [], []
        fold_train_pearsons, fold_best_iters, fold_times = [], [], []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            t0 = time.time()
            try:
                pearson, r2, mse, best_iter, train_pearson = \
                    self._train_and_evaluate_with_model(
                        params, X_train, y_train, X_val, y_val, early_stopping_rounds
                    )
            except Exception as e:
                log.red(f"Trial {trial.number} Fold {fold_idx} failed: {e}")
                raise optuna.exceptions.TrialPruned()
            fold_time = time.time() - t0
            fold_pearsons.append(pearson)
            fold_r2s.append(r2)
            fold_mses.append(mse)
            fold_train_pearsons.append(train_pearson)
            fold_best_iters.append(best_iter)
            fold_times.append(fold_time)
            # early prune: OverfitRatio<0.3 means extreme overfitting
            if fold_idx >= 1 and len(fold_pearsons) >= 2:
                early_val_p = np.mean(fold_pearsons)
                early_train_p = np.mean(fold_train_pearsons)
                if early_train_p > 0.01:
                    early_ratio = early_val_p / early_train_p
                    if early_ratio < 0.3:
                        log.yellow(f"Trial {trial.number} early prune at fold {fold_idx}: "
                                   f"ValPearson={early_val_p:.4f} TrainPearson={early_train_p:.4f} "
                                   f"OverfitRatio={early_ratio:.3f} < 0.3")
                        raise optuna.exceptions.TrialPruned()

        # weighted average: last fold closest to test window gets higher weight
        n_folds = len(fold_pearsons)
        if n_folds == 3:
            fold_weights = np.array([0.25, 0.30, 0.45])
        elif n_folds == 5:
            fold_weights = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
        else:
            fold_weights = np.ones(n_folds) / n_folds
        avg_pearson = np.average(fold_pearsons, weights=fold_weights)
        avg_r2 = np.average(fold_r2s, weights=fold_weights)
        avg_mse = np.average(fold_mses, weights=fold_weights)
        avg_train_pearson = np.average(fold_train_pearsons, weights=fold_weights)
        avg_best_iter = np.mean(fold_best_iters)
        total_time = np.sum(fold_times)
        overfitting_ratio = avg_pearson / avg_train_pearson if avg_train_pearson > 0 else 0

        trial_log = {
            'trial_number': trial.number,
            'params': params,
            'avg_pearson': avg_pearson,
            'avg_r2': avg_r2,
            'avg_mse': avg_mse,
            'avg_train_pearson': avg_train_pearson,
            'avg_best_iteration': avg_best_iter,
            'overfitting_ratio': overfitting_ratio,
            'total_time': total_time,
            'fold_pearsons': fold_pearsons,
            'fold_r2s': fold_r2s,
            'fold_mses': fold_mses,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        self.trial_logs.append(trial_log)
        self._update_top5(trial_log)

        last_fold_ratio = fold_pearsons[-1] / avg_pearson if avg_pearson > 0 else 0
        fold_detail = " | ".join(f"F{i}={p:.4f}" for i, p in enumerate(fold_pearsons))
        log.inf(f"Trial {trial.number}: Pearson={avg_pearson:.4f} R2={avg_r2:.5f} MSE={avg_mse:.6f} "
                f"TrainPearson={avg_train_pearson:.4f} OverfitRatio={overfitting_ratio:.3f} "
                f"BestIter={avg_best_iter:.0f} Time={total_time:.1f}s")
        log.inf(f"  Fold detail: {fold_detail} | LastFold/Mean={last_fold_ratio:.3f}")
        return avg_pearson

    def _update_top5(self, trial_log):
        self.top5_models.append({
            'trial_number': trial_log['trial_number'],
            'params': trial_log['params'],
            'avg_pearson': trial_log['avg_pearson'],
            'avg_r2': trial_log['avg_r2'],
            'avg_mse': trial_log['avg_mse'],
        })
        self.top5_models.sort(key=lambda x: x['avg_pearson'], reverse=True)
        self.top5_models = self.top5_models[:5]

    def tune(self, X, y, n_trials=100, algorithm='bayesian', n_splits=5,
             early_stopping_rounds=50, timeout_per_trial=1800, study_name=None,
             generate_plots=False, sample_ratio=1.0, day_boundaries=None):
        if study_name is None:
            study_name = f'gbt_tuning_{algorithm}'
        # day-subsample: pick contiguous days, done once for all trials
        if sample_ratio < 1.0 and day_boundaries is not None:
            n_days = len(day_boundaries)
            n_keep = int(n_days * sample_ratio)
            max_start_day = n_days - n_keep
            max_offset = min(max_start_day, int(n_days * 0.2))
            rng = np.random.RandomState(42)
            start_day = rng.randint(0, max_offset + 1)
            keep_days = day_boundaries[start_day:start_day + n_keep]
            indices = np.concatenate([np.arange(s, e) for s, e in keep_days])
            X = X[indices]
            y = y[indices]
            log.inf(f"Day-subsample: start_day={start_day}, keep {n_keep} of {n_days} days, "
                    f"{len(indices)} rows (ratio={sample_ratio})")

        log.inf(f"Starting auto-tuning (MeowModel fit->predict->eval): "
                f"algorithm={algorithm}, n_trials={n_trials}, "
                f"n_splits={n_splits}, early_stopping={early_stopping_rounds}, "
                f"timeout_per_trial={timeout_per_trial}s, sample_ratio={sample_ratio}")
        sampler = TPESampler(seed=42) if algorithm == 'bayesian' else RandomSampler(seed=42)
        db_path = os.path.join(self.output_dir, f'{study_name}.db')
        storage = f'sqlite:///{db_path}'
        study = optuna.create_study(
            study_name=study_name, sampler=sampler, storage=storage,
            direction='maximize',
            load_if_exists=True,
        )

        def objective_with_timeout(trial):
            import threading
            result_container = [None]
            error_container = [None]
            def run_objective():
                try:
                    result_container[0] = self._objective(
                        trial, X, y, n_splits, early_stopping_rounds
                    )
                except Exception as e:
                    error_container[0] = e
            thread = threading.Thread(target=run_objective)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout_per_trial)
            if thread.is_alive():
                log.yellow(f"Trial {trial.number} timed out after {timeout_per_trial}s, pruning")
                raise optuna.exceptions.TrialPruned()
            if error_container[0] is not None:
                raise error_container[0]
            return result_container[0]

        t_start = time.time()
        study.optimize(objective_with_timeout, n_trials=n_trials)
        t_total = time.time() - t_start
        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value
        log.inf(f"Tuning completed in {t_total:.1f}s")
        log.inf(f"Best Pearson: {best_value:.4f}")
        log.inf(f"Best params: {json.dumps(best_params, indent=2)}")
        self._save_results(study, best_params, best_value, t_total, algorithm)
        if generate_plots:
            self._generate_visualization(study)
        else:
            log.inf("Skipping visualization (generate_plots=False)")
        self._generate_report(study, best_params, best_value, t_total, algorithm)
        self._generate_reproducible_script(best_params)
        return best_params

    def _save_results(self, study, best_params, best_value, total_time, algorithm):
        json_path = os.path.join(self.output_dir, 'best_params.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)
        log.inf(f"Best params saved to {json_path}")
        results = {
            'best_params': best_params,
            'best_pearson': float(best_value) if isinstance(best_value, (np.integer, np.floating)) else best_value,
            'total_time': total_time,
            'algorithm': algorithm,
            'n_trials': len(study.trials),
            'top5_models': [],
        }
        # serialize numpy types for JSON
        serializable_logs = []
        for tl in self.trial_logs:
            d = dict(tl)
            d['params'] = {k: (float(v) if isinstance(v, (np.integer, np.floating)) else v)
                          for k, v in d['params'].items()}
            for key in ['avg_pearson', 'avg_r2', 'avg_mse', 'avg_train_pearson',
                       'avg_best_iteration', 'overfitting_ratio', 'total_time']:
                if key in d and isinstance(d[key], (np.integer, np.floating)):
                    d[key] = float(d[key])
            for key in ['fold_pearsons', 'fold_r2s', 'fold_mses']:
                if key in d:
                    d[key] = [float(v) for v in d[key]]
            serializable_logs.append(d)
        results['trial_logs'] = serializable_logs
        for m in self.top5_models:
            m2 = dict(m)
            m2['params'] = {k: (float(v) if isinstance(v, (np.integer, np.floating)) else v)
                           for k, v in m2['params'].items()}
            for key in ['avg_pearson', 'avg_r2', 'avg_mse']:
                if key in m2 and isinstance(m2[key], (np.integer, np.floating)):
                    m2[key] = float(m2[key])
            results['top5_models'].append(m2)
        all_results_path = os.path.join(self.output_dir, 'all_results.json')
        with open(all_results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        log.inf(f"All results saved to {all_results_path}")
        for i, m in enumerate(self.top5_models):
            log.inf(f"Top{i+1} model: Pearson={m['avg_pearson']:.4f}, params={m['params']}")

    def _generate_visualization(self, study):
        try:
            import matplotlib
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            import matplotlib.pyplot as plt
            fig_dir = os.path.join(self.output_dir, 'figures')
            os.makedirs(fig_dir, exist_ok=True)
            trials = study.trials
            trial_numbers = [t.number for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            trial_values = [t.value for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            if trial_numbers:
                plt.figure(figsize=(12, 6))
                plt.plot(trial_numbers, trial_values, 'b-', alpha=0.5, label='Pearson')
                best_so_far = []
                current_best = -np.inf
                for v in trial_values:
                    current_best = max(current_best, v)
                    best_so_far.append(current_best)
                plt.plot(trial_numbers, best_so_far, 'r-', linewidth=2, label='Best Pearson')
                plt.xlabel('Trial')
                plt.ylabel('Pearson')
                plt.title('Pearson vs Trial Number')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(fig_dir, 'optimization_history.png'), dpi=200, bbox_inches='tight')
                plt.close()
                log.inf("Optimization history plot saved")
            try:
                importance = optuna.importance.get_param_importances(study)
                if importance:
                    plt.figure(figsize=(10, 6))
                    params_names = list(importance.keys())
                    params_values = list(importance.values())
                    y_pos = np.arange(len(params_names))
                    plt.barh(y_pos, params_values, align='center', alpha=0.8, color='steelblue')
                    plt.yticks(y_pos, params_names)
                    plt.xlabel('Importance')
                    plt.title('Hyperparameter Importance')
                    for i, v in enumerate(params_values):
                        plt.text(v, i, f' {v:.4f}', va='center', fontsize=9)
                    plt.tight_layout()
                    plt.savefig(os.path.join(fig_dir, 'param_importance.png'), dpi=200, bbox_inches='tight')
                    plt.close()
                    log.inf("Parameter importance plot saved")
            except Exception as e:
                log.yellow(f"Could not generate param importance: {e}")
            try:
                if len(trials) > 1:
                    ax = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
                    if hasattr(ax, 'figure'):
                        ax.figure.savefig(os.path.join(fig_dir, 'parallel_coordinate.png'), dpi=200, bbox_inches='tight')
                    import matplotlib.pyplot as plt2
                    plt2.close('all')
                    log.inf("Parallel coordinate plot saved")
            except Exception as e:
                log.yellow(f"Could not generate parallel coordinate: {e}")
        except ImportError:
            log.yellow("matplotlib not available, skipping visualization")

    def _generate_report(self, study, best_params, best_value, total_time, algorithm):
        r = []
        r.append("# LightGBM Auto-Tuning Report\n")
        r.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        r.append("## 1. Tuning Summary\n")
        r.append(f"- Algorithm: {algorithm}\n")
        r.append(f"- Total Trials: {len(study.trials)}\n")
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        r.append(f"- Completed Trials: {completed}\n")
        r.append(f"- Total Time: {total_time:.1f}s ({total_time/3600:.2f}h)\n")
        r.append(f"- Best Pearson: {best_value:.4f}\n\n")
        r.append("## 2. Best Hyperparameters\n")
        r.append("```json\n")
        r.append(json.dumps(best_params, indent=2, ensure_ascii=False))
        r.append("\n```\n\n")
        r.append("## 3. Performance Improvement\n")
        baseline = 0.0576
        imp = best_value - baseline
        imp_pct = (imp / baseline) * 100 if baseline > 0 else 0
        r.append(f"- Baseline Pearson: {baseline:.4f}\n")
        r.append(f"- Optimized Pearson: {best_value:.4f}\n")
        r.append(f"- Improvement: {imp:+.4f} ({imp_pct:+.1f}%)\n")
        r.append(f"- Target Pearson: 0.09\n")
        r.append(f"- Target Reached: {'Yes' if best_value >= 0.09 else 'No'}\n\n")
        r.append("## 4. Top 5 Models\n")
        r.append("| Rank | Pearson | R2 | MSE | Key Params |\n")
        r.append("|------|---------|-----|-----|------------|\n")
        for i, m in enumerate(self.top5_models):
            kp = f"depth={m['params'].get('max_depth')}, lr={m['params'].get('learning_rate')}, leaves={m['params'].get('num_leaves')}"
            r.append(f"| {i+1} | {m['avg_pearson']:.4f} | {m['avg_r2']:.5f} | {m['avg_mse']:.6f} | {kp} |\n")
        r.append("\n")
        r.append("## 5. Parameter Importance\n")
        try:
            importance = optuna.importance.get_param_importances(study)
            r.append("| Parameter | Importance |\n")
            r.append("|-----------|------------|\n")
            for param, imp_val in importance.items():
                r.append(f"| {param} | {imp_val:.4f} |\n")
        except Exception as e:
            r.append(f"Could not compute: {e}\n")
        r.append("\n")
        r.append("## 6. Overfitting Analysis\n")
        if self.trial_logs:
            best_log = max(self.trial_logs, key=lambda x: x['avg_pearson'])
            ratio = best_log.get('overfitting_ratio', 0)
            r.append(f"- Best trial overfitting ratio (val_pearson/train_pearson): {ratio:.3f}\n")
            r.append(f"- Overfitting check (ratio >= 0.9): {'Pass' if ratio >= 0.9 else 'Warning'}\n")
            r.append(f"- Train Pearson: {best_log.get('avg_train_pearson', 0):.4f}\n")
            r.append(f"- Val Pearson: {best_log.get('avg_pearson', 0):.4f}\n")
        r.append("\n")
        r.append("## 7. How to Apply Best Params\n")
        r.append("```python\n")
        r.append("from mdl import MeowModel\n")
        r.append("model = MeowModel(cacheDir=None)\n")
        r.append("model.load_params_from_json('tuning_output/best_params.json')\n")
        r.append("# or manually:\n")
        r.append("model.update_params({\n")
        for k, v in best_params.items():
            r.append(f"    '{k}': {v},\n")
        r.append("})\n")
        r.append("model.fit(xdf, ydf)\n")
        r.append("```\n\n")
        r.append("## 8. Trial Details\n")
        r.append("| Trial | Pearson | R2 | MSE | TrainPearson | OverfitRatio | BestIter | Time(s) |\n")
        r.append("|-------|---------|-----|-----|-------------|-------------|----------|--------|\n")
        for tl in self.trial_logs[:30]:
            r.append(f"| {tl['trial_number']} | {tl['avg_pearson']:.4f} | {tl['avg_r2']:.5f} | "
                    f"{tl['avg_mse']:.6f} | {tl['avg_train_pearson']:.4f} | "
                    f"{tl['overfitting_ratio']:.3f} | {tl['avg_best_iteration']:.0f} | "
                    f"{tl['total_time']:.1f} |\n")
        if len(self.trial_logs) > 30:
            r.append(f"| ... | | | | | | | | (first 30 of {len(self.trial_logs)} trials) |\n")
        r.append("\n")
        report_path = os.path.join(self.output_dir, 'tuning_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(r)
        log.inf(f"Report saved to {report_path}")

    def _generate_reproducible_script(self, best_params):
        s = []
        s.append('"""\nReproducible training script with best hyperparameters\n')
        s.append(f'Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        s.append('Uses project MeowModel with update_params() to apply optimized params.\n"""\n\n')
        s.append('from mdl import MeowModel\n')
        s.append('from eval import MeowEvaluator\n\n')
        s.append('BEST_PARAMS = ')
        s.append(json.dumps(best_params, indent=2, ensure_ascii=False))
        s.append('\n\n')
        s.append("""def train_best_model(xdf, ydf):
    model = MeowModel(cacheDir=None)
    model.update_params(BEST_PARAMS)
    model.update_params({'early_stopping_rounds': 50})
    model.fit(xdf, ydf)
    return model

if __name__ == '__main__':
    # Load your data using project modules
    # from dl import MeowDataLoader
    # from feat import MeowFeatureGenerator
    # ... load and generate features ...
    # model = train_best_model(xdf, ydf)
    print("Load data and call train_best_model(xdf, ydf)")
""")
        script_path = os.path.join(self.output_dir, 'train_best.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.writelines(s)
        log.inf(f"Reproducible script saved to {script_path}")
