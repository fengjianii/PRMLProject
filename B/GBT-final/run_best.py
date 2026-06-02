from meow import MeowEngine

engine = MeowEngine(h5dir="../../archive", cacheDir=None)
engine.model.load_params_from_json('tuning_output/best_params.json')
engine.fit(20230601, 20231130)
engine.eval(20231201, 20231229)