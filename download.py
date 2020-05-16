from tensorflow.keras.datasets import cifar100
import pickle
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
data = (X_train, y_train), (X_test, y_test)
pickle.dump(data, open('data.p', 'wb'))
