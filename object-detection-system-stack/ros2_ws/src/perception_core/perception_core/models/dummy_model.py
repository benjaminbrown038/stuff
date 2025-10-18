import random

class DummyModel:
    """Simulates an object detection model."""

    def __init__(self, model_path=None):
        # model_path is ignored for this dummy
        self.model_path = model_path

    def predict(self, image):
        """
        Simulate predictions.
        Returns a list of bounding boxes and class names.
        """
        predictions = []
        for i in range(3):  # Simulate 3 objects per frame
            predictions.append({
                'id': i,
                'x': random.uniform(0.1, 0.8),
                'y': random.uniform(0.1, 0.8),
                'width': 0.1,
                'height': 0.1,
                'confidence': random.uniform(0.5, 1.0),
                'class_name': f"object_{i}"
            })
        return predictions
