import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from classes.KNN import KNN
from classes.NaiveBayes import NaiveBayes
from classes.ID3 import ID3
from classes.ModelLoader import ModelLoader

def generate_data():
    X_train = pd.DataFrame({
        'Feature1': [10, 20, 30, 40, 50],
        'Feature2': [60, 50, 40, 30, 20],
        'Feature3': [70, 80, 90, 100, 110],
        'Feature4': [5, 15, 25, 35, 45],
    })

    y_train = pd.Series(["type 1", "type 0", "type 1", "type 0", "type 1"])
    
    X_test = pd.DataFrame({
        'Feature1': [15, 25, 35],
        'Feature2': [55, 45, 35],
        'Feature3': [85, 95, 105],
        'Feature4': [10, 20, 30],
    })
    
    return X_train, y_train, X_test

X_train, y_train, X_test = generate_data()

def test_knn():
    knn = KNN(k=5)
    
    knn.fit(X_train, y_train)
    ModelLoader.save(knn, 'test_knn_model.pkl')
    loaded_oknn = ModelLoader.load('test_knn_model.pkl')
    
    predictions = loaded_oknn.predict(X_test)
    print(f"KNN Predictions: {predictions}")

def test_sklearn_knn():
    sklearn_knn = KNeighborsClassifier(n_neighbors=5)
    sklearn_knn.fit(X_train, y_train)
    
    predictions = sklearn_knn.predict(X_test)
    print(f"Sklearn KNN Predictions: {predictions}")

def test_naive_bayes():
    nb = NaiveBayes()
    
    nb.fit(X_train, y_train)
    ModelLoader.save(nb, 'test_naive_bayes_model.pkl')
    loaded_nb = ModelLoader.load('test_naive_bayes_model.pkl')
    
    predictions = loaded_nb.predict(X_test)
    print(f"Naive Bayes Predictions: {predictions}")

def test_sklearn_naive_bayes():
    sklearn_nb = GaussianNB()
    sklearn_nb.fit(X_train, y_train)
    
    predictions = sklearn_nb.predict(X_test)
    print(f"Sklearn Naive Bayes Predictions: {predictions}")

def test_id3():
    id3 = ID3()
    
    id3.fit(X_train, y_train)
    ModelLoader.save(id3, 'test_id3_model.pkl')
    loaded_id3 = ModelLoader.load('test_id3_model.pkl')
    
    predictions = loaded_id3.predict(X_test)
    print(f"ID3 Predictions: {predictions}")
    
from sklearn.tree import DecisionTreeClassifier

def test_sklearn_decision_tree_entropy():
    sklearn_dt = DecisionTreeClassifier(criterion='entropy')
    sklearn_dt.fit(X_train, y_train)
    
    predictions = sklearn_dt.predict(X_test)
    print(f"Sklearn Decision Tree (Entropy) Predictions: {predictions}")
    

def printSeparator():
    print()
    print("=" * 60)
    print()

if __name__ == '__main__':
    printSeparator()
    print("Testing KNN Model...")
    test_knn()
    print("\nTesting Sklearn KNN Model...")
    test_sklearn_knn()
    
    printSeparator()
    print("\nTesting Naive Bayes Model...")
    test_naive_bayes()
    print("\nTesting Sklearn Naive Bayes Model...")
    test_sklearn_naive_bayes()
    
    printSeparator()
    print("\nTesting ID3 Model...")
    test_id3()
    print("\nTesting Sklearn Decision Tree (Entropy)...")
    test_sklearn_decision_tree_entropy()