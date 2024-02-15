import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

def custom_label_encoder(text_label):
    return final_status_mapping.get(text_label, 0)

def custom_label_encoder_direction(text_label):
    return label_mapping.get(text_label, -1)

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x > 0) * 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binary_cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-9  # To prevent log(0)
    return -np.sum(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

def forward_pass(x, W, b):
    out = [x]
    in_hidden = np.dot(W[1], out[0]) + b[1]
    out_hidden = relu(in_hidden)
    out.append(out_hidden)
    in_output = np.dot(W[2], out[1]) + b[2]
    out_output = sigmoid(in_output)
    out.append(out_output)
    return out

def backward_pass(out, W, b, y_true):
    grad_W = [None] * len(W)
    grad_b = [None] * len(b)
    grad_out = out[-1] - y_true
    grad_W[2] = np.outer(grad_out, out[1])
    grad_b[2] = grad_out
    grad_in_hidden = np.dot(W[2].T, grad_out) * relu_prime(out[1])
    grad_W[1] = np.outer(grad_in_hidden, out[0])
    grad_b[1] = grad_in_hidden
    return grad_W, grad_b

def update_parameters(W, b, grad_W, grad_b, learning_rate):
    for k in range(1, len(W)):
        W[k] -= learning_rate * grad_W[k]
        b[k] -= learning_rate * grad_b[k]
    return W, b

def preprocess_input(df, text_label_columns):
    for col in text_label_columns:
        df[col] = df[col].apply(custom_label_encoder_direction)
    return df.values.astype(float)

def train(X_df, Y, W, b, epochs, learning_rate, text_label_columns):
    Y_encoded = np.array([custom_label_encoder(label) for label in Y]).reshape(-1, 1)
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X_df)):
            sample = X_df.iloc[i]
            x_processed = preprocess_input(pd.DataFrame([sample]), text_label_columns)
            y_true_encoded = Y_encoded[i]
            out = forward_pass(x_processed[0], W, b)
            y_pred = out[-1]
            loss = binary_cross_entropy_loss(y_true_encoded, y_pred)
            total_loss += loss
            grad_W, grad_b = backward_pass(out, W, b, y_true_encoded)
            W, b = update_parameters(W, b, grad_W, grad_b, learning_rate)
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(X_df)}")
    return W, b

def probability_to_label(probability):
    return 'RESCUED' if probability > 0.5 else 'CAUGHT'

def test(X_df, Y, W, b, text_label_columns):
    Y_encoded = np.array([custom_label_encoder(label) for label in Y]).reshape(-1, 1)
    total_loss = 0
    all_true_labels = []
    all_predicted_labels = []
    correct_predictions = 0

    for i in range(len(X_df)):
        sample = X_df.iloc[i]
        x_processed = preprocess_input(pd.DataFrame([sample]), text_label_columns)
        y_true_encoded = Y_encoded[i]
        out = forward_pass(x_processed[0], W, b)
        y_pred = out[-1]
        loss = binary_cross_entropy_loss(y_true_encoded, y_pred)
        total_loss += loss

        predicted_label = probability_to_label(y_pred[0])
        true_label = Y.iloc[i]

        all_true_labels.append(true_label)
        all_predicted_labels.append(predicted_label)

        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / len(Y)
    cm = confusion_matrix(all_true_labels, all_predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predicted_labels, labels=['CAUGHT', 'RESCUED'])

    return total_loss / len(X_df), accuracy, precision, recall, f1, cm

final_status_mapping = {'CAUGHT': 0, 'RESCUED': 1}
label_mapping = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3, 'STAY': 4, 'UNREACHABLE': -1}

df = pd.read_csv("m2-test-final.csv")
df = df.dropna()
X_df = df.drop(columns=['action_taken', 'final_result'])
text_label_columns = []
Y = df['final_result']

X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y, test_size=0.2, random_state=0)

learning_rates = [0.001, 0.0001, 0.0005]
epoch_numbers = [100, 200, 300]
hidden_layer_sizes = [100, 200]
output_size = 1
best_combination = {'learning_rate': 0, 'epoch': 0, 'hidden_layer_size': 0, 'average_f1_accuracy': 0, 'best_cm': None}

for learning_rate in learning_rates:
    for epochs in epoch_numbers:
        for hidden_layer_size in hidden_layer_sizes:
            W = [None, np.random.randn(hidden_layer_size, X_train.shape[1]), np.random.randn(output_size, hidden_layer_size)]
            b = [None, np.random.randn(hidden_layer_size), np.random.randn(output_size)]

            W, b = train(X_train, Y_train, W, b, epochs, learning_rate, text_label_columns)
            test_loss, test_accuracy, precision, recall, f1, cm = test(X_test, Y_test, W, b, text_label_columns)
            f1 = np.mean(f1)
            precision = np.mean(precision)
            recall = np.mean(recall)

            print(f"Learning Rate: {learning_rate}, Epochs: {epochs}, Hidden Layer Size: {hidden_layer_size}")
            print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

            average_f1_accuracy = (test_accuracy + f1) / 2
            if average_f1_accuracy > best_combination['average_f1_accuracy']:
                best_combination = {
                    'learning_rate': learning_rate,
                    'epoch': epochs,
                    'hidden_layer_size': hidden_layer_size,
                    'average_f1_accuracy': average_f1_accuracy,
                    'best_cm': cm
                }

print("Best Combination:")
print(f"Learning Rate: {best_combination['learning_rate']}, Epochs: {best_combination['epoch']}, Hidden Layer Size: {best_combination['hidden_layer_size']}")
print(f"Average of F1 Score and Accuracy: {best_combination['average_f1_accuracy']}")

sns.heatmap(best_combination['best_cm'], annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Best Model')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
