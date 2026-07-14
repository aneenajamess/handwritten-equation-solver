import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Flatten,Dense

(x_train,y_train),(x_test,y_test)=mnist.load_data()


x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)

model=Sequential([
    Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation="relu",
        input_shape=(28,28,1)
    ),
    MaxPooling2D(pool_size=(2, 2)),

      Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(
        units=128,
        activation="relu"
    ),

    Dense(
        units=10,
        activation="softmax"
    )
])

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history=model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(x_test,y_test)
)

loss,accuracy=model.evaluate(
    x_test,
    y_test,

)

print(f"\nTest Accuracy: {accuracy:.4f}")

# -----------------------------
# Save Model
# -----------------------------
model.save("digit_model.keras")

print("\nModel saved successfully as digit_model.keras")

