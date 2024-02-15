import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

def custom_label_encoder(text_label):
    return label_mapping.get(text_label, 4)

def custom_label_encoder_direction(text_label):
    return label_mapping.get(text_label, -1)

one_hot_encoder = OneHotEncoder(sparse=False, categories=[range(5)])
one_hot_encoder.fit(np.array(range(5)).reshape(-1, 1))

def one_hot_encode_labels(labels):
    integer_encoded = np.array([custom_label_encoder(label) for label in labels]).reshape(-1, 1)
    return one_hot_encoder.transform(integer_encoded)

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x > 0) * 1

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9))

def dropout(x, dropout_rate):
    mask = np.random.binomial(n=1, p=1-dropout_rate, size=x.shape)
    return x * mask

def forward_pass(x, W, b, dropout_rate=0):
    out = [x]
    for layer in range(1, len(W)-1):
        activation = relu(np.dot(W[layer], out[layer-1]) + b[layer])
        if dropout_rate > 0:
            activation = dropout(activation, dropout_rate)
        out.append(activation)
    out.append(softmax(np.dot(W[-1], out[-1]) + b[-1]))
    return out

def backward_pass(out, W, b, y_true):
    grad_W = [None] * len(W)
    grad_b = [None] * len(b)
    grad_out = out[-1] - y_true
    grad_W[-1] = np.outer(grad_out, out[-2])
    grad_b[-1] = grad_out
    for layer in range(len(W) - 2, 0, -1):
        grad_out = np.dot(W[layer + 1].T, grad_out) * relu_prime(out[layer])
        grad_W[layer] = np.outer(grad_out, out[layer - 1])
        grad_b[layer] = grad_out
    return grad_W, grad_b

def update_parameters(W, b, grad_W, grad_b, learning_rate):
    for k in range(1, len(W)):
        W[k] -= learning_rate * grad_W[k]
        b[k] -= learning_rate * grad_b[k]
    return W, b

def preprocess_input(df, text_label_columns):
    for col in text_label_columns:
        df[col] = df[col].apply(custom_label_encoder_direction)
    processed_data = df.values.astype(float)
    return processed_data

def one_hot_to_label(one_hot_vector):
    label_index = np.argmax(one_hot_vector)
    inverse_label_mapping = {v: k for k, v in label_mapping_final.items()}
    return inverse_label_mapping.get(label_index, 'STAY')

def calculate_precision_recall(confusion_mat):
    class_count = confusion_mat.shape[0]
    precision = np.zeros(class_count)
    recall = np.zeros(class_count)

    for i in range(class_count):
        true_positives = confusion_mat[i, i]
        predicted_positives = np.sum(confusion_mat[:, i])
        actual_positives = np.sum(confusion_mat[i, :])

        precision[i] = true_positives / predicted_positives if predicted_positives else 0
        recall[i] = true_positives / actual_positives if actual_positives else 0

    return precision, recall

def test_with_predictions(X_df, Y, W, b, text_label_columns):
    Y_encoded = one_hot_encode_labels(Y)
    total_loss = 0
    correct_predictions = 0
    all_true_labels = []
    all_predicted_labels = []

    for i in range(len(X_df)):
        sample = X_df.iloc[i]
        x_processed = preprocess_input(pd.DataFrame([sample]), text_label_columns)
        y_true_encoded = Y_encoded[i]
        out = forward_pass(x_processed[0], W, b)
        y_pred = out[-1]
        loss = cross_entropy_loss(y_true_encoded, y_pred)
        total_loss += loss

        predicted_label = one_hot_to_label(y_pred)
        true_label = Y.iloc[i]

        all_true_labels.append(true_label)
        all_predicted_labels.append(predicted_label)

        if predicted_label == true_label:
            correct_predictions += 1

    average_loss = total_loss / len(X_df)
    accuracy = correct_predictions / len(X_df)
    return average_loss, accuracy, all_true_labels, all_predicted_labels

def train(X_df, Y, W, b, epochs, learning_rate, text_label_columns, dropout_rate):
    Y_encoded = one_hot_encode_labels(Y)
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X_df)):
            sample = X_df.iloc[i]
            x_processed = preprocess_input(pd.DataFrame([sample]), text_label_columns)
            y_true_encoded = Y_encoded[i]
            out = forward_pass(x_processed[0], W, b, dropout_rate)
            y_pred = out[-1]
            loss = cross_entropy_loss(y_true_encoded, y_pred)
            total_loss += loss
            grad_W, grad_b = backward_pass(out, W, b, y_true_encoded)
            W, b = update_parameters(W, b, grad_W, grad_b, learning_rate)
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(X_df)}")
    return W, b

label_mapping = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3, 'STAY': 4, 'UNREACHABLE': -1}
label_mapping_final = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3, 'STAY': 4}

df = pd.read_csv("vtest.csv")
X_df = df.drop(columns=['action_taken', 'actual_crew_pos_x', 'actual_crew_pos_y','next_position_x', 'next_position_y', 'final_result'])
text_label_columns = []
Y = df['action_taken']

X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y, test_size=0.2, random_state=0)

learning_rates = [0.0001, 0.0005, 0.001]
epoch_numbers = [100,200,300]
hidden_layer_sizes = [50, 100, 150]
output_size = 5
dropout_rate = 0.3

best_combination = {'learning_rate': 0, 'epoch': 0, 'hidden_layer_size': 0, 'average_f1_accuracy': 0}

for learning_rate in learning_rates:
    for epochs in epoch_numbers:
        for hidden_layer_size in hidden_layer_sizes:
            W = [None, np.random.randn(hidden_layer_size, X_train.shape[1]), np.random.randn(output_size, hidden_layer_size)]
            b = [None, np.random.randn(hidden_layer_size), np.random.randn(output_size)]

            W, b = train(X_train, Y_train, W, b, epochs, learning_rate, text_label_columns, dropout_rate)

            test_loss, test_accuracy, true_labels, predicted_labels = test_with_predictions(X_test, Y_test, W, b, text_label_columns)

            cm = confusion_matrix(true_labels, predicted_labels, labels=['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'])
            precision, recall = calculate_precision_recall(cm)
            average_precision = np.mean(precision)
            average_recall = np.mean(recall)
            f1 = f1_score(true_labels, predicted_labels, average='weighted')

            print(f"Learning Rate: {learning_rate}, Epochs: {epochs}, Hidden Layer Size: {hidden_layer_size}")
            print(f"Test Accuracy: {test_accuracy}, Average Precision: {average_precision}, Average Recall: {average_recall}, F1 Score: {f1}")
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.show()

            average_f1_accuracy = (test_accuracy + f1) / 2
            if average_f1_accuracy > best_combination['average_f1_accuracy']:
                best_combination = {
                    'learning_rate': learning_rate,
                    'epoch': epochs,
                    'hidden_layer_size': hidden_layer_size,
                    'average_f1_accuracy': average_f1_accuracy
                }

print("Best Combination:")
print(f"Learning Rate: {best_combination['learning_rate']}, Epochs: {best_combination['epoch']}, Hidden Layer Size: {best_combination['hidden_layer_size']}")
print(f"Average of F1 Score and Accuracy: {best_combination['average_f1_accuracy']}")
