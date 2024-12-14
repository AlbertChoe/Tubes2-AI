import pandas as pd
from KNN import KNN, OptimizedKNN
from NaiveBayes import NaiveBayes
from ID3 import ID3
from ModelLoader import ModelLoader

def generate_data():
    X_train = pd.DataFrame({
        'Feature1': [10, 20, 30, 40, 50],
        'Feature2': [60, 50, 40, 30, 20],
        'Feature3': [70, 80, 90, 100, 110],
        'Feature4': [5, 15, 25, 35, 45],
    })

    y_train = pd.Series([1, 0, 1, 0, 1])
    
    X_test = pd.DataFrame({
        'Feature1': [15, 25, 35],
        'Feature2': [55, 45, 35],
        'Feature3': [85, 95, 105],
        'Feature4': [10, 20, 30],
    })
    
    y_test = pd.Series([1, 0, 1])
    
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = generate_data()

def test_knn():
    knn = KNN(k=5)
    
    knn.fit(X_train, y_train)
    ModelLoader.save(knn, 'knn_model.pkl')
    loaded_knn = ModelLoader.load('knn_model.pkl')
    
    predictions = loaded_knn.predict(X_test)
    print(f"KNN Predictions: {predictions}")
    
def test_oknn():
    oknn = OptimizedKNN(k=5)
    
    oknn.fit(X_train, y_train)
    ModelLoader.save(oknn, 'oknn_model.pkl')
    loaded_knn = ModelLoader.load('oknn_model.pkl')
    
    predictions = loaded_knn.predict(X_test)
    print(f"OptimizedKNN Predictions: {predictions}")


def test_naive_bayes():
    nb = NaiveBayes()
    
    nb.fit(X_train, y_train)
    ModelLoader.save(nb, 'naive_bayes_model.pkl')
    loaded_nb = ModelLoader.load('naive_bayes_model.pkl')
    
    predictions = loaded_nb.predict(X_test)
    print(f"Naive Bayes Predictions: {predictions}")


def test_id3():
    id3 = ID3()
    
    id3.fit(X_train, y_train)
    ModelLoader.save(id3, 'id3_model.pkl')
    loaded_id3 = ModelLoader.load('id3_model.pkl')
    
    predictions = loaded_id3.predict(X_test)
    print(f"ID3 Predictions: {predictions}")


if __name__ == '__main__':
    print("Testing KNN Model...")
    test_knn()
    print("Testing Optimized KNN Model...")
    test_oknn()
    print("\nTesting Naive Bayes Model...")
    test_naive_bayes()
    print("\nTesting ID3 Model...")
    test_id3()