import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Đọc dữ liệu
bodyswing_df = pd.read_csv("BODYSWING.txt")
handswing_df = pd.read_csv("HANDSWING.txt")
clapswing_df = pd.read_csv("CLAPSWING.txt")

X = []
y = []
no_of_timesteps = 10

dataset = handswing_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(0)

dataset = bodyswing_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(1)



dataset = clapswing_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(2)


X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

# Thay đổi cấu trúc của mô hình để phù hợp với nhiều nhãn
num_labels = 3  # Số lượng nhãn mới
model  = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = num_labels, activation="softmax"))  # Sử dụng softmax và số lượng units tương ứng với số lượng nhãn
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "categorical_crossentropy")  # Sử dụng categorical_crossentropy


from keras.utils import to_categorical

# Chuyển đổi nhãn thành dạng one-hot encoding
y_categorical = to_categorical(y, num_classes=num_labels)

# Chia dữ liệu thành train và test
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Tiến hành huấn luyện mô hình
model.fit(X_train, y_train, epochs=16, batch_size=32,validation_data=(X_test, y_test))
model.save("model.h5")


# Đánh giá mô hình trên tập kiểm tra
y_pred = np.argmax(model.predict(X_test), axis=1)
y_test_argmax = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_argmax, y_pred)
precision = precision_score(y_test_argmax, y_pred, average='weighted')
recall = recall_score(y_test_argmax, y_pred, average='weighted')
f1 = f1_score(y_test_argmax, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test_argmax, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix: \n{conf_matrix}')