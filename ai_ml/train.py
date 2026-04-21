from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib
import os

# Dataset: [IQ, CGPA] → Placement (0 = No, 1 = Yes)
X = np.array([
    [110, 6.5],
    [120, 7.0],
    [130, 8.0],
    [90, 5.5],
    [100, 6.0],
    [140, 9.0],
    [150, 9.5],
    [85, 5.0],
    [95, 5.8],
    [135, 8.5]
])

y = np.array([0, 0, 1, 0, 0, 1, 1, 0, 0, 1])

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
os.makedirs("/opt/ml/model", exist_ok=True)
joblib.dump(model, "/opt/ml/model/model.joblib")