import os
from log import log
from dl import MeowDataLoader
from feat import MeowFeatureGenerator
from mdl import MeowModel
from eval import MeowEvaluator
from tradingcalendar import Calendar
from backtest import format_summary, run_from_predictions


class MeowEngine(object):
    def __init__(self, h5dir, cacheDir):
        self.calendar = Calendar()
        self.h5dir = h5dir
        if not os.path.exists(h5dir):
            raise ValueError("Data directory not exists: {}".format(self.h5dir))
        if not os.path.isdir(h5dir):
            raise ValueError("Invalid data directory: {}".format(self.h5dir))
        self.cacheDir = cacheDir # this is not used in sample code
        self.dloader = MeowDataLoader(h5dir=h5dir)
        self.featGenerator = MeowFeatureGenerator(cacheDir=cacheDir)
        self.model = MeowModel(cacheDir=cacheDir)
        self.evaluator = MeowEvaluator(cacheDir=cacheDir)

    def fit(self, startDate, endDate):
        dates = self.calendar.range(startDate, endDate)
        rawData = self.dloader.loadDates(dates)
        log.inf("Running model fitting...")
        xdf, ydf = self.featGenerator.genFeatures(rawData)
        self.model.fit(xdf, ydf)

    def predict(self, xdf):
        return self.model.predict(xdf)

    def eval(self, startDate, endDate, predictionPath=None, agentSummaryPath=None):
        log.inf("Running model evaluation...")
        dates = self.calendar.range(startDate, endDate)
        rawData = self.dloader.loadDates(dates)
        xdf, ydf = self.featGenerator.genFeatures(rawData)
        ydf.loc[:, "forecast"] = self.predict(xdf)
        metrics = self.evaluator.eval(ydf)
        if predictionPath is not None:
            self.savePredictions(ydf, predictionPath)
        if agentSummaryPath is not None:
            self.runAgentBacktest(ydf, agentSummaryPath)
        return metrics

    def savePredictions(self, ydf, predictionPath):
        os.makedirs(os.path.dirname(predictionPath) or ".", exist_ok=True)
        ydf.reset_index().to_csv(predictionPath, index=False)
        log.inf("Saved predictions to {}".format(predictionPath))

    def runAgentBacktest(self, ydf, agentSummaryPath):
        _, summary = run_from_predictions(ydf.reset_index(), cost_bps=1.0)
        os.makedirs(os.path.dirname(agentSummaryPath) or ".", exist_ok=True)
        with open(agentSummaryPath, "w") as f:
            f.write(format_summary(summary))
            f.write("\n")
        log.inf("Saved Agent backtest summary to {}".format(agentSummaryPath))
        log.inf(format_summary(summary))


if __name__ == "__main__":
    import os
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    engine = MeowEngine(h5dir=data_dir, cacheDir=None)
    engine.fit(20230601, 20231130)
    prediction_path = os.environ.get("MEOW_PREDICTION_OUTPUT")
    agent_summary_path = os.environ.get("MEOW_AGENT_SUMMARY_OUTPUT")
    engine.eval(
        20231201,
        20231229,
        predictionPath=prediction_path,
        agentSummaryPath=agent_summary_path,
    )
