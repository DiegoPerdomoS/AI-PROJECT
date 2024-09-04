import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb

warnings.filterwarnings("ignore", category=FutureWarning)

# Cargar los datos
columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df = pd.read_csv('car.data', names=columns)

# Reemplazar variables categóricas por códigos numéricos
replacement_dict = {
    "buying": {"vhigh": 3, "high": 2, "med": 1, "low": 0},
    "maint": {"vhigh": 3, "high": 2, "med": 1, "low": 0},
    "doors": {"2": 0, "3": 1, "4": 2, "5more": 3},
    "persons": {"2": 0, "4": 1, "more": 2},
    "lug_boot": {"small": 0, "med": 1, "big": 2},
    "safety": {"low": 0, "med": 1, "high": 2},
    "class": {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}
}

df.replace(replacement_dict, inplace=True)

# Ingeniería de características
df['buying_maint'] = df['buying'] * df['maint']
df['safety_squared'] = df['safety'] ** 2
df['persons_lug_boot'] = df['persons'] * df['lug_boot']

# División del conjunto de datos
X = df[["buying", "maint", "doors", "persons", "lug_boot", "safety", 
        "buying_maint", "safety_squared", "persons_lug_boot"]]
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Escalado de las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir los parámetros del modelo
params = {
    'objective': 'multi:softmax',
    'num_class': 4,
    'reg_alpha': 1,
    'reg_lambda': 1,
    'eta': 0.1,
    'eval_metric': ['mlogloss', 'merror']  # Log loss and classification error
}

# Modelo XGBoost con Regularización
model = xgb.XGBClassifier(**params)

# Track metrics during training
eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]

# Entrenar el modelo y almacenar los resultados de evaluación
model.fit(X_train_scaled, y_train, eval_set=eval_set, verbose=False)

# Extraer la historia de evaluación
results = model.evals_result()

# Graficar la pérdida de entrenamiento y prueba por epoch
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

plt.figure(figsize=(12, 6))
plt.plot(x_axis, results['validation_0']['mlogloss'], label='Train Loss')
plt.plot(x_axis, results['validation_1']['mlogloss'], label='Test Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.title('Train and Test Loss over Epochs')
plt.grid(True)
plt.show()

# Graficar la precisión de entrenamiento y prueba por epoch
plt.figure(figsize=(12, 6))
plt.plot(x_axis, 1 - np.array(results['validation_0']['merror']), label='Train Accuracy')
plt.plot(x_axis, 1 - np.array(results['validation_1']['merror']), label='Test Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy over Epochs')
plt.grid(True)
plt.show()

# Predicción y evaluación final
train_predictions = model.predict(X_train_scaled)
test_predictions = model.predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print(f"Final Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, test_predictions)

# Graficar la matriz de confusión
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(y)))
plt.xticks(tick_marks, np.unique(y))
plt.yticks(tick_marks, np.unique(y))

# Añadir etiquetas
thresh = conf_matrix.max() / 2
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, f"{conf_matrix[i, j]}", 
             horizontalalignment="center", 
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()
