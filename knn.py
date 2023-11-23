from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# تعریف داده‌ها و برچسب‌ها
X = np.array([[1, 1], [1, 2], [2, 2], [4, 4], [5, 5], [6, 6]])
y = np.array([0, 0, 0, 1, 1, 1])

# تعریف مقادیر مختلف برای K
k_values = list(range(1, 10))

# نگهداری بهترین مقدار K
best_k = None
best_accuracy = 0.0

# ارزیابی مدل با استفاده از ارزیابی متقابل
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X, y, cv=3)  # تعداد بخش‌ها برای ارزیابی متقابل: 3
    accuracy = np.mean(scores)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print("بهترین مقدار K:", best_k)
print("دقت بهترین مقدار K:", best_accuracy)