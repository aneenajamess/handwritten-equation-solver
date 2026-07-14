import pandas as pd
import cv2
import numpy as np 

#prepare symbol data
label_df=pd.read_csv("HASYv2/hasy-data-labels.csv")

wanted_symbols = ["+","-","/","\\equiv","\\times"]

selected=label_df[label_df["latex"].isin(wanted_symbols)]

label_map={"+":0,"-":1,"/":2,"\\equiv":3,"\\times":4}

images=[]
labels=[]

for index,row in selected.iterrows():
    path="HASYv2/" +row["path"]

    img=cv2.imread(path)

    if img is None:
        continue

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    resized=cv2.resize(gray,(28,28))

    normalized=resized/255.0

    images.append(normalized)

    label=label_map[row["latex"]]
    labels.append(label)

X=np.array(images)
y=np.array(labels)

X=X.reshape(X.shape[0],28,28,1)

#train data
from sklearn.model_selection import train_test_split

X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#cnn architecture
from tensorflow.keras.layers import (Conv2D,MaxPooling2D,Flatten,Dense)
from tensorflow.keras.models import Sequential

model=Sequential()

model.add(
    Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation="relu",
        input_shape=(28,28,1)
    )
)
model.add(
    MaxPooling2D(
        pool_size=(2,2)
    )
)

model.add(
    Conv2D(
        64,
        (3,3),
        activation="relu"
    )
)

model.add(
    MaxPooling2D((2,2))
)


model.add(
    Flatten()
)


model.add(
    Dense(
        64,
        activation="relu"
    )
)


model.add(
    Dense(
        5,
        activation="softmax"
    )
)

model.summary()

#compiling
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)

print(test_accuracy)
model.save(
    "models/operator_model.keras"
)