import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
import csv
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

# Open Data File and assemble training data
with open('data.csv', 'r') as file:
    reader = csv.reader(file)
    x = []
    y = []
    for n, row in enumerate(reader):
        if n == 0:
            continue
        if [float(i) for i in row[:3]]:
            x.append([float(i) for i in row[:3]])
            y.append(row[4])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=43)

# Train SVM
clf = svm.SVC()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

errors = []

# Calculate Error
for real, pred in zip(y_test, y_pred):
    dv = abs(float(real)-float(pred))
    errors.append(dv/float(real))

error = np.nansum(errors) / len(errors)

nonan_test = []
nonan_pred = []

for test, pred in zip(y_test, y_pred):
    t = float(test)
    p = float(pred)
    if abs(t) < 100000 and abs(p) < 100000:
        nonan_test.append(t)
        nonan_pred.append(p)

print("Accuracy:", 100 - error)

X = range(len(y_pred))

plt.style.use('dark_background')
fig, ax = plt.subplots()
plt.xlabel("Samples")
plt.ylabel("Density [1/cm^3]")

ax.plot(X, [float(i) for i in y_pred], linewidth=1.0, label='Predicted', color="#5F54F5") # ORANGE 
ax.plot(X, [float(i) for i in y_test], linewidth=1.0, label='Real', color="#D36114") # BLUE

plt.legend()
plt.show()

# # Create a convolutional neural network
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(3,1)),
#   tf.keras.layers.Dense(3, activation='relu'),
#   tf.keras.layers.Dense(1, activation='softmax')
# ])

# # Train neural network
# model.compile(
#     optimizer="adam",
#     loss=tf.keras.losses.categorical_hinge(),
#     metrics=["accuracy"]
# )
# model.fit(x_train, y_train, epochs=10)

# # Evaluate neural network performance
# model.evaluate(x_test,  y_test, verbose=2)

# # Save model to file
# if len(sys.argv) == 2:
#     filename = sys.argv[1]
#     model.save(filename)
#     print(f"Model saved to {filename}.")