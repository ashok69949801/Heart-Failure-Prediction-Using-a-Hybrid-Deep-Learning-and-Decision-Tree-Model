import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pickle


data = pd.read_csv("C:/Users/Seshu Nagu/OneDrive/Desktop/heart.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


num_classes = len(np.unique(y_encoded))


if num_classes > 2:
    from tensorflow.keras.utils import to_categorical
    y_encoded = to_categorical(y_encoded)


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


input_shape = X_train.shape[1]
ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(input_shape,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1 if num_classes == 2 else num_classes, activation='sigmoid' if num_classes == 2 else 'softmax')
])


loss_function = 'binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy'
ann_model.compile(optimizer=Adam(learning_rate=0.001), loss=loss_function, metrics=['accuracy'])


history = ann_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1, validation_data=(X_test, y_test))


penultimate_layer = ann_model.layers[-2]
weights, biases = penultimate_layer.get_weights()


input_feature_names = data.columns[:-1]
feature_names = []
num_features = min(weights.shape[1], len(input_feature_names))

for i in range(weights.shape[1]):
    if i < num_features:
        top_contributors_indices = np.argsort(-np.abs(weights[:, i]))[:3]
        top_contributors_indices = np.clip(top_contributors_indices, 0, len(input_feature_names) - 1)
        top_contributors = [input_feature_names[idx] for idx in top_contributors_indices]
        feature_name = f"Feature_{i+1} ({', '.join(top_contributors)})"
        feature_names.append(feature_name)


feature_extractor = Sequential(ann_model.layers[:-1])
X_train_features = feature_extractor.predict(X_train)
X_test_features = feature_extractor.predict(X_test)

decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train_features, np.argmax(y_train, axis=1) if num_classes > 2 else y_train)


y_pred_dt = decision_tree.predict(X_test_features)


decision_tree_accuracy = accuracy_score(np.argmax(y_test, axis=1) if num_classes > 2 else y_test, y_pred_dt)
print(f"\nDecision Tree Accuracy: {decision_tree_accuracy * 100:.2f}%")


ann_model.save('ann_model.h5')
with open('dt_model.pkl', 'wb') as f:
    pickle.dump(decision_tree, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Models and scaler saved successfully!")


cm = confusion_matrix(np.argmax(y_test, axis=1) if num_classes > 2 else y_test, y_pred_dt)
print("\nConfusion Matrix:")
print(cm)


report = classification_report(np.argmax(y_test, axis=1) if num_classes > 2 else y_test, y_pred_dt,
                              target_names=label_encoder.classes_ if num_classes > 2 else None)
print("\nClassification Report:")
print(report)


correct_predictions = np.sum(np.diag(cm))
incorrect_predictions = np.sum(cm) - correct_predictions

print("\nDetailed Accuracy Breakdown:")
print(f"Total Test Samples: {len(y_test)}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Incorrect Predictions: {incorrect_predictions}")
print(f"Decision Tree Accuracy: {decision_tree_accuracy * 100:.2f}%")


plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='dashed')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('ANN Training & Validation Loss')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='dashed')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('ANN Training & Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


feature_importances = decision_tree.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance After ANN Feature Extraction')
plt.gca().invert_yaxis()
plt.show()


correlation_matrix = data.iloc[:, :-1].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True)
plt.title('Feature Correlation Heatmap')
plt.show()


print("Training completed and models saved to 'ann_model.h5', 'dt_model.pkl', and 'scaler.pkl'")