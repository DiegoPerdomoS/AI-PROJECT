# IMPROVED ORIGINAL

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
df = df.sample(frac=1, random_state=10).reset_index(drop=True)

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

alfa = 0.4
max_epochs = 3000

# Cross-validation
for fold in range(k_folds):
    start, end = fold * fold_size, (fold + 1) * fold_size
    
    val_x, val_y = df_x[start:end], df_y[start:end]
    train_x = np.concatenate([df_x[:start], df_x[end:]], axis=0)
    train_y = np.concatenate([df_y[:start], df_y[end:]], axis=0)
    
    params_list = np.random.uniform(-0.01, 0.01, (num_classes, train_x.shape[1]))
    
    train_accuracies = {i: [] for i in range(num_classes)}
    val_accuracies = {i: [] for i in range(num_classes)}
    test_accuracies = {i: [] for i in range(num_classes)} 
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
            test_acc = calculate_accuracy(params_list, df_x, df_y, current_class) 
            
            train_accuracies[current_class].append(train_acc)
            val_accuracies[current_class].append(val_acc)
            test_accuracies[current_class].append(test_acc)
        
        avg_epoch_cost = np.mean(epoch_costs)
        epoch_errors.append(avg_epoch_cost)

        # Print accuracy every 500 epochs
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}:")
            for current_class in range(num_classes):
                print(f"  Class {current_class}: Train Accuracy = {train_accuracies[current_class][-1]:.4f}, Validation Accuracy = {val_accuracies[current_class][-1]:.4f}, Test Accuracy = {test_accuracies[current_class][-1]:.4f}")

    if fold == k_folds - 1:
        for current_class in range(num_classes):
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))

            # Accuracy plot
            axs[0].plot(train_accuracies[current_class], label=f'Train Accuracy Class {current_class}', linestyle='--')
            axs[0].plot(val_accuracies[current_class], label=f'Validation Accuracy Class {current_class}')
            axs[0].plot(test_accuracies[current_class], label=f'Test Accuracy Class {current_class}')  # Test accuracy over epochs
            axs[0].set_xlabel('Epochs')
            axs[0].set_ylabel('Accuracy')
            axs[0].set_title(f'Accuracy over Epochs for Class {current_class} (Fold {fold + 1})')
            axs[0].legend()

            # Cost plot
            axs[1].plot(costs_per_class[current_class], label=f'Cost for Class {current_class}')
            axs[1].set_xlabel('Epochs')
            axs[1].set_ylabel('Cost')
            axs[1].set_title(f'Cost over Epochs for Class {current_class} (Fold {fold + 1})')
            axs[1].legend()

            plt.tight_layout()
            plt.show()

final_test_accuracy = calculate_accuracy(params_list, df_x, df_y)
final_validation_accuracy = calculate_accuracy(params_list, val_x, val_y)

print(f"Final Test Accuracy: {final_test_accuracy * 100:.2f}%")
print(f"Final Validation Accuracy: {final_validation_accuracy * 100:.2f}%")
