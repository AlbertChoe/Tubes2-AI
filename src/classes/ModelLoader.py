import pickle

class ModelLoader:
    @staticmethod
    def save(model: object, filename: str) -> None:
        """Save the model to a .pkl file."""
        filename = "pkl_model/" + filename
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {filename}")

    @staticmethod
    def load(filename: str) -> object:
        """Load the model from a .pkl file."""
        filename = "pkl_model/" + filename
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {filename}")
        return model