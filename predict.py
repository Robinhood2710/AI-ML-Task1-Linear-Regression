import joblib
import numpy as np

# Load Saved Model
model = joblib.load("model.pkl")

# Sample House Data
sample = np.array([
    [8.3252,41,6.984,1.024,322,2.555,37.88,-122.23]
])

# Prediction
prediction = model.predict(sample)

print("Predicted House Price:")
print(prediction)