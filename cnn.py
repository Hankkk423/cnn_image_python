import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model


### Load and Preprocess Data ###
dataset_name = 'cifar10'
(train_data, test_data), info = tfds.load(dataset_name, split=['train', 'test'], with_info=True, as_supervised=True)

# Iterate through the first 5 samples in the training data
for i, (image, label) in enumerate(train_data.take(3)):
    print(f"Sample {i+1}:")
    print("Image shape:", image.shape)
    print("Label:", label.numpy())
    print("Pixel values:")
    print(image.numpy())
    print("=" * 30)

# Get the size of the training data
train_data_size = tf.data.experimental.cardinality(train_data).numpy()
# Get the size of the test data
test_data_size = tf.data.experimental.cardinality(test_data).numpy()

print("Size of training data:", train_data_size)
print("Size of test data:", test_data_size)


### Preprocess Images ###
def preprocess_image(image, label):
    image = tf.image.resize(image, (64, 64))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label
5
train_data = train_data.map(preprocess_image)
test_data = test_data.map(preprocess_image)

batch_size = 16
train_data = train_data.batch(batch_size).shuffle(buffer_size=1000)
test_data = test_data.batch(batch_size)


### Build CNN Model ###
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])


### Train the Model ###
num_epochs = 5
decay_steps = (train_data_size/batch_size) * 4
initial_learning_rate = 0.001

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=decay_steps, decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, epochs=num_epochs, validation_data=test_data)


### Evaluate and Visualize ###
test_loss, test_acc = model.evaluate(test_data)
print(f"Test accuracy: {test_acc}")

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
# Save the plot as an image
plt.savefig('training_history.png')  # Specify the filename and format (e.g., .png)
# Display the plot
plt.show()


### Prediction ###
print('Make Prediction: ')
predictions = model.predict(test_data)

# # Iterate through the predictions for each test sample
# for i, pred_probs in enumerate(predictions):
#     # Find the index with the highest probability (predicted class)
#     predicted_class = tf.argmax(pred_probs).numpy()

#     # Get the true label from the test_data
#     true_label = next(iter(test_data.unbatch().skip(i).take(1)))[1].numpy()

#     # Get the class names from the dataset info
#     class_names = info.features['label'].names

#     # Print the results
#     print(f"Sample {i+1}:")
#     print(f"True label: {class_names[true_label]}")
#     print(f"Predicted class: {class_names[predicted_class]}")
#     print(f"Predicted probabilities: {pred_probs}")
#     print("=" * 30)


### Save the model to a file ###
model.save('saved-model/my_model.h5')
print('Model Saved!')


### Load the saved model ###
loaded_model = load_model('saved-model/my_model.h5')

# Evaluate the loaded model on test data
loaded_test_loss, loaded_test_acc = loaded_model.evaluate(test_data)
print(f"Loaded model test accuracy: {loaded_test_acc}")



