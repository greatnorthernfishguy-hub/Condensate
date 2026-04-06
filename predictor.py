"""Condensate Predictor — delegates to Rust RustPredictor."""
import condensate_core


class Predictor:
    def __init__(self):
        self._predictor = condensate_core.RustPredictor()

    def learn(self, graph_builder):
        """Learn from a GraphBuilder's inner AccessGraph."""
        graph = graph_builder.inner if hasattr(graph_builder, 'inner') else graph_builder
        self._predictor.learn(graph)

    def predict(self, path, top_k=10):
        return self._predictor.predict(path, top_k)

    def score(self, events):
        return self._predictor.score(events)

    def is_learned(self):
        return self._predictor.is_learned()
