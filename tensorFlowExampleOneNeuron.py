
#  pip install tensorflow

# ------------------------------------------------------------
# TENSORFLOW: Learning a Simple Maths Pattern
# ------------------------------------------------------------
# This program trains a neural network (just 1 neuron!)
# to learn the relationship between x and y.
#
# The relationship is:        y = 3x + 1
#
# We do NOT tell the AI this formula.
# Instead, we give it examples and it works it out itself.
# ------------------------------------------------------------

import tensorflow as tf
import numpy as np


# ------------------------------------------------------------
# 1. TRAINING DATA
# ------------------------------------------------------------
# xs = inputs (x values)
# ys = outputs (correct y values)
#
# The pattern is: y = 3x + 1
# But the AI does not know this!
# ------------------------------------------------------------

xs = np.array([0, 1, 2, 3, 4, 5], dtype=float)
ys = np.array([1, 4, 7, 10, 13, 16], dtype=float)


# ------------------------------------------------------------
# 2. BUILD THE MODEL
# ------------------------------------------------------------
# We create a Sequential model (layers in a line).
#
# The model has:
#   - an Input layer (1 number goes in – the x value)
#   - a Dense layer with 1 neuron (1 number comes out – the predicted y value)
#
# Inside that 1 neuron, TensorFlow will learn:
#   - a weight (W)
#   - a bias (B)
#
# The neuron calculates: prediction = (W * x) + B
# ------------------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),      # The model will receive 1 input number
    tf.keras.layers.Dense(units=1)   # One neuron producing one output
])


# ------------------------------------------------------------
# 3. COMPILE THE MODEL
# ------------------------------------------------------------
# This sets up the training process.
#
# optimizer='sgd'            → how the model adjusts W and B (gradient descent)
# loss='mean_squared_error'  → how the model measures how wrong it is
# ------------------------------------------------------------

model.compile(
    optimizer='sgd',
    loss='mean_squared_error'
)


# ------------------------------------------------------------
# 4. TRAIN (FIT) THE MODEL
# ------------------------------------------------------------
# epochs = number of training cycles.
# Each epoch:
#   - the model makes predictions
#   - compares them with the correct answers
#   - adjusts W and B to reduce the error
# ------------------------------------------------------------

model.fit(xs, ys, epochs=500, verbose=False)

print("Training complete!")
print("----------------------------------------")


# ------------------------------------------------------------
# 5. USE THE MODEL TO MAKE A PREDICTION
# ------------------------------------------------------------
# We ask the AI: what is y when x = 10?
#
# IMPORTANT:
# Modern TensorFlow requires NumPy arrays with shape (samples, features)
# So we use [[10.0]] (1 sample, 1 input feature).
# ------------------------------------------------------------

x_new = np.array([[10.0]], dtype=float)
prediction = model.predict(x_new)

print("Prediction for x = 10:")
print(prediction)
print("----------------------------------------")


# ------------------------------------------------------------
# 6. OPTIONAL: SHOW THE LEARNED WEIGHT AND BIAS
# ------------------------------------------------------------
# You can reveal the learned equation!
# It should be very close to y = 3x + 1.
# ------------------------------------------------------------

W, B = model.layers[0].get_weights()
print(f"Learned weight (W): {W[0][0]}")
print(f"Learned bias   (B): {B[0]}")

# ------------------------------------------------------------
# END OF PROGRAM
# ------------------------------------------------------------
