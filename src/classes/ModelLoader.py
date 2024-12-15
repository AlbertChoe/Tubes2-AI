import pickle

class ModelLoader:
    @staticmethod
    def save(model: object, filename: str):
        """
        Save the model to a .pkl file.

        Parameters:
        ----------
            model (object): The model object to save.
            filename (str): The name of the file to save the model to. The file will be saved in the 'pkl_model/' directory.
        """
        filename = "pkl_model/" + filename
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {filename}")

    @staticmethod
    def load(filename: str) -> object:
        """
        Load the model from a .pkl file.

        Parameters:
        ----------
            filename (str): The name of the file to load the model from. The file should be located in the 'pkl_model/' directory.

        Returns:
        --------
            object: The model object loaded from the file.
        """
        filename = "pkl_model/" + filename
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {filename}")
        return model