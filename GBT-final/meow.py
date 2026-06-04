import os
import numpy as np
from log import log
from dl import MeowDataLoader
from feat import MeowFeatureGenerator
from mdl import MeowModel
from eval import MeowEvaluator
from tradingcalendar import Calendar


class MeowEngine(object):

    def __init__(self, h5dir, cacheDir, feature_set='full'):
        self.calendar = Calendar()
        self.h5dir = h5dir
        if not os.path.exists(h5dir):
            raise ValueError("Data directory not exists: {}".format(self.h5dir))
        if not os.path.isdir(h5dir):
            raise ValueError("Invalid data directory: {}".format(self.h5dir))
        self.cacheDir = cacheDir
        self.dloader = MeowDataLoader(h5dir=h5dir)
        self.featGenerator = MeowFeatureGenerator(cacheDir=cacheDir, feature_set=feature_set)
        self.model = MeowModel(cacheDir=cacheDir)
        self.evaluator = MeowEvaluator(cacheDir=cacheDir)

    def fit(self, startDate, endDate):
        dates = self.calendar.range(startDate, endDate)
        rawData = self.dloader.loadDates(dates)
        log.inf("Running model fitting...")
        xdf, ydf = self.featGenerator.genFeatures(rawData)
        self.model.fit(xdf, ydf)
        try:
            log.inf("生成训练曲线...")
            figure_path = self.model.plot_training_curves(output_dir='figures')
            if figure_path:
                log.inf(f"训练曲线已保存到: {figure_path}")
            else:
                log.yellow("无法生成训练曲线")
        except Exception as e:
            log.yellow(f"生成训练曲线时出错: {e}")

    def predict(self, xdf):
        return self.model.predict(xdf)

    def eval(self, startDate, endDate):
        log.inf("Running model evaluation...")
        dates = self.calendar.range(startDate, endDate)
        rawData = self.dloader.loadDates(dates)
        xdf, ydf = self.featGenerator.genFeatures(rawData)
        predictions = self.predict(xdf)
        ydf.loc[:, "forecast"] = predictions
        self.evaluator.eval(ydf)
        try:
            log.inf("生成预测可视化图表...")
            y_true = ydf["fret12"].to_numpy()
            y_pred = predictions
            generated_files = self.model.create_training_report(
                y_true=y_true,
                y_pred=y_pred,
                output_dir='figures'
            )
            if generated_files:
                log.inf(f"生成了 {len(generated_files)} 个可视化图表:")
                for file_path in generated_files:
                    log.inf(f"  - {file_path}")
            else:
                log.yellow("无法生成可视化图表")
        except Exception as e:
            log.yellow(f"生成可视化图表时出错: {e}")

    def tune(self, startDate, endDate, n_trials=100, algorithm='bayesian',
             n_splits=5, early_stopping_rounds=50, timeout_per_trial=1800,
             output_dir='tuning_output', apply_best=True, sample_ratio=0.3,
             generate_plots=False):
        from tuner import MeowTuner
        dates = self.calendar.range(startDate, endDate)
        rawData = self.dloader.loadDates(dates)
        log.inf("Preparing data for auto-tuning...")
        xdf, ydf = self.featGenerator.genFeatures(rawData)
        # extract per-day row boundaries for day-based subsampling
        date_values = xdf.index.get_level_values('date').to_numpy()
        unique_dates = np.unique(date_values)
        day_boundaries = []
        for d in unique_dates:
            mask = date_values == d
            indices = np.where(mask)[0]
            day_boundaries.append((indices[0], indices[-1] + 1))
        X = xdf.to_numpy().astype(np.float32)
        y = ydf.to_numpy().ravel().astype(np.float32)
        log.inf(f"Tuning data shape: X={X.shape}, y={y.shape}, n_days={len(unique_dates)}")
        tuner = MeowTuner(output_dir=output_dir)
        best_params = tuner.tune(
            X, y,
            day_boundaries=day_boundaries,
            n_trials=n_trials,
            algorithm=algorithm,
            n_splits=n_splits,
            early_stopping_rounds=early_stopping_rounds,
            timeout_per_trial=timeout_per_trial,
            generate_plots=generate_plots,
            sample_ratio=sample_ratio,
        )
        if apply_best:
            log.inf("Applying best params to current model...")
            self.model.update_params(best_params)
            # early_stopping scales with lr (same as during tuning)
            lr = best_params.get('learning_rate', 0.01)
            dynamic_esr = max(50, int(0.1 / lr * 50))
            self.model.update_params({'early_stopping_rounds': dynamic_esr})
            log.inf(f"Best params applied (early_stopping={dynamic_esr}). Call engine.fit() to train.")
        return best_params


if __name__ == "__main__":
    engine = MeowEngine(h5dir="../../archive", cacheDir=None)
    engine.fit(20230601, 20231130)
    engine.eval(20231201, 20231229)
