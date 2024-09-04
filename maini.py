import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# Load and preprocess data
columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df = pd.read_csv('car.data', names=columns)

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

df['buying_maint'] = df['buying'] * df['maint']
df['safety_squared'] = df['safety'] ** 2
df['persons_lug_boot'] = df['persons'] * df['lug_boot']

# Shuffle
df = df.sample(frac=1, random_state=40).reset_index(drop=True)

df_x = df[["buying", "maint", "doors", "persons", "lug_boot", "safety", 
           "buying_maint", "safety_squared", "persons_lug_boot"]]
df_y = df["class"]

# Add bias term
df_x = np.c_[np.ones(len(df_x)), df_x]
df_y = np.array(df_y)

num_classes = len(np.unique(df_y))
k_folds = 5
fold_size = len(df_x) // k_folds

# Logistic regression functions
def h(params, sample):
    z = -np.dot(params, sample)
    return 1 / (1 + np.exp(z))

def cost_function(params, samples, y, current_class):
    targets = (y == current_class).astype(int)
    predictions = h(params, samples.T)
    cost = -targets * np.log(predictions) - (1 - targets) * np.log(1 - predictions)
    return np.mean(cost)

def GD(params, samples, y, alfa, current_class):
    targets = (y == current_class).astype(int)
    predictions = h(params, samples.T)
    errors = predictions - targets
    gradient = np.dot(errors, samples) / len(samples)
    params -= alfa * gradient
    
    cost = cost_function(params, samples, y, current_class)
    return params, cost

def predict_class(params_list, sample):
    probabilities = h(params_list, sample.T)
    return np.argmax(probabilities, axis=0)

def calculate_accuracy(params_list, samples, labels, current_class=None):
    predictions = predict_class(params_list, samples)
    if current_class is None:
        return np.mean(predictions == labels)
    else:
        relevant_samples = labels == current_class
        if not relevant_samples.any():
            return 0
        return np.mean(predictions[relevant_samples] == labels[relevant_samples])

alfa = 0.3
max_epochs = 2000

# Cross-validation
cross_val_train_accuracies = []
cross_val_val_accuracies = []

for fold in range(k_folds):
    start, end = fold * fold_size, (fold + 1) * fold_size
    
    val_x, val_y = df_x[start:end], df_y[start:end]
    train_x = np.concatenate([df_x[:start], df_x[end:]], axis=0)
    train_y = np.concatenate([df_y[:start], df_y[end:]], axis=0)
    
    params_list = np.random.uniform(-0.01, 0.01, (num_classes, train_x.shape[1]))
    
    train_accuracies = {i: [] for i in range(num_classes)}
    val_accuracies = {i: [] for i in range(num_classes)}
    costs_per_class = {i: [] for i in range(num_classes)}
    epoch_errors = []

    for epoch in range(max_epochs):
        epoch_costs = []
        for current_class in range(num_classes):
            params_list[current_class], class_cost = GD(params_list[current_class], train_x, train_y, alfa, current_class)
            epoch_costs.append(class_cost)
            costs_per_class[current_class].append(class_cost)
            
            train_acc = calculate_accuracy(params_list, train_x, train_y, current_class)
            val_acc = calculate_accuracy(params_list, val_x, val_y, current_class)
            
            train_accuracies[current_class].append(train_acc)
            val_accuracies[current_class].append(val_acc)
        
        avg_epoch_cost = np.mean(epoch_costs)
        epoch_errors.append(avg_epoch_cost)

    cross_val_train_accuracies.append(train_accuracies)
    cross_val_val_accuracies.append(val_accuracies)
    
    # Only graph the last fold
    if fold == k_folds - 1:
        # Graph accuracy and cost for each class
        for current_class in range(num_classes):
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))
            
            # Accuracy 
            axs[0].plot(train_accuracies[current_class], label=f'Train Accuracy Class {current_class}', linestyle='--')
            axs[0].plot(val_accuracies[current_class], label=f'Validation Accuracy Class {current_class}')
            axs[0].set_xlabel('Epochs')
            axs[0].set_ylabel('Accuracy')
            axs[0].set_title(f'Accuracy over Epochs for Class {current_class} (Fold {fold + 1})')
            axs[0].legend()

            # Cost 
            axs[1].plot(costs_per_class[current_class], label=f'Cost for Class {current_class}')
            axs[1].set_xlabel('Epochs')
            axs[1].set_ylabel('Cost')
            axs[1].set_title(f'Cost over Epochs for Class {current_class} (Fold {fold + 1})')
            axs[1].legend()

            plt.tight_layout()
            plt.show()

# Retrain on the entire training set
params_list = np.random.uniform(-0.01, 0.01, (num_classes, train_x.shape[1]))

for epoch in range(max_epochs):
    for current_class in range(num_classes):
        params_list[current_class], _ = GD(params_list[current_class], train_x, train_y, alfa, current_class)

# Evaluate on the test set
final_test_accuracy = calculate_accuracy(params_list, df_x, df_y)
print(f"Final Overall Accuracy (Cross-Validation): {final_test_accuracy * 100:.2f}%")

# Confusion Matrix
confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

for i in range(len(df_y)):
    true_label = df_y[i]
    pred_label = predict_class(params_list, df_x[i])
    confusion_matrix[true_label, pred_label] += 1

plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, [0, 1, 2, 3]) 
plt.yticks(tick_marks, [0, 1, 2, 3])

thresh = confusion_matrix.max() / 2.0
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
