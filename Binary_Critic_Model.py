import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

def custom_label_encoder(text_label):
    return final_status_mapping.get(text_label, 0)

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x > 0) * 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binary_cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))

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

def preprocess_input(df):
    processed_data = df.values.astype(float)
    return processed_data

def train(X_df, Y, W, b, epochs, learning_rate):
    Y_encoded = np.array([custom_label_encoder(label) for label in Y]).reshape(-1, 1)
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X_df)):
            sample = X_df.iloc[i]
            x_processed = preprocess_input(pd.DataFrame([sample]))
            y_true_encoded = Y_encoded[i]
            out = forward_pass(x_processed[0], W, b)
            y_pred = out[-1]
            loss = binary_cross_entropy_loss(y_true_encoded, y_pred)
            total_loss += loss
            grad_W, grad_b = backward_pass(out, W, b, y_true_encoded)
            W, b = update_parameters(W, b, grad_W, grad_b, learning_rate)
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(X_df)}, Learning Rate: {learning_rate}")
    return W, b

def probability_to_label(probability):
    return 'RESCUED' if probability > 0.5 else 'CAUGHT'

def test(X_df, Y, W, b):
    Y_encoded = np.array([custom_label_encoder(label) for label in Y]).reshape(-1, 1)
    total_loss = 0
    all_true_labels = []
    all_predicted_labels = []

    for i in range(len(X_df)):
        sample = X_df.iloc[i]
        x_processed = preprocess_input(pd.DataFrame([sample]))
        y_true_encoded = Y_encoded[i]
        out = forward_pass(x_processed[0], W, b)
        y_pred = out[-1]
        loss = binary_cross_entropy_loss(y_true_encoded, y_pred)
        total_loss += loss

        predicted_label = 1 if y_pred > 0.5 else 0
        true_label = Y_encoded[i][0]

        all_true_labels.append(true_label)
        all_predicted_labels.append(predicted_label)

    average_loss = total_loss / len(X_df)
    accuracy = np.mean(np.array(all_true_labels) == np.array(all_predicted_labels))

    precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predicted_labels, average='binary')

    cm = confusion_matrix(all_true_labels, all_predicted_labels)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return average_loss, accuracy, precision, recall, f1

final_status_mapping = {'CAUGHT': 0, 'RESCUED': 1}

caught_df=pd.read_csv("Balanced_Actor_caught.csv")
rescued_df=pd.read_csv("Balanced_Actor_rescued.csv")
caught_df1=pd.read_csv("Balanced_Actor_caught1.csv")
rescued_df1=pd.read_csv("Balanced_Actor_rescued1.csv")
caught_df2=pd.read_csv("Balanced_Actor_caught2.csv")
rescued_df2=pd.read_csv("Balanced_Actor_rescued2.csv")
caught_df3=pd.read_csv("Balanced_Actor_caught3.csv")
rescued_df3=pd.read_csv("Balanced_Actor_rescued3.csv")

caught_combined = pd.concat([caught_df, caught_df1, caught_df2, caught_df3], ignore_index=True)


rescued_combined = pd.concat([rescued_df, rescued_df1, rescued_df2, rescued_df3], ignore_index=True)

if len(rescued_combined) > len(caught_combined):
    rescued_combined = rescued_combined[:len(caught_combined)]



df = pd.concat([rescued_combined, caught_combined])
df['Unnamed: 0'] = df['Unnamed: 0']%150
df = df.dropna()
print(df)
X_df = df.drop(columns=['action_taken', 'final_result'])
Y = df['final_result']

X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y, test_size=0.2, random_state=0)

input_size = X_train.shape[1]
hidden_layer_size = 200
output_size = 1 
epochs=2

W = [None, np.random.randn(hidden_layer_size, input_size), np.random.randn(output_size, hidden_layer_size)]
b = [None, np.random.randn(hidden_layer_size), np.random.randn(output_size)]

W, b = train(X_train, Y_train, W, b, epochs, learning_rate=0.001)

test_loss, test_accuracy, average_precision, average_recall, f1 = test(X_test, Y_test, W, b)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Average Precision: {average_precision}, Average Recall: {average_recall}, F1 Score: {f1}")

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


def forward_pass1(x, W, b):
    out = [x]
    in_hidden = np.dot(W[1], out[0]) + b[1]
    out_hidden = relu(in_hidden)
    out.append(out_hidden)
    in_output = np.dot(W[2], out[1]) + b[2]
    out_output = sigmoid(in_output)
    out.append(out_output)
    return out

def predict_percent_rescued_for_row(sample, sample_dropped, W, b, text_label_columns):
    
    action=sample['action_taken']
    x_processed = sample_dropped
    out = forward_pass1(x_processed, W, b)
    y_pred = out[-1]
    return y_pred[0]

columns = ['Unnamed: 0', 'bot_position_x', 'bot_position_y', 'actual_crew_pos_x',
           'actual_crew_pos_y', 'up_x', 'up_y', 'down_x', 'down_y', 'left_x',
           'left_y', 'right_x', 'right_y', 'up_alien_prob', 'up_crew_prob',
           'down_alien_prob', 'down_crew_prob', 'left_alien_prob',
           'left_crew_prob', 'right_alien_prob', 'right_crew_prob',
           'max_prob_cell_x', 'max_prob_cell_y', 'next_position_x',
           'next_position_y', 'final_result']

unique_values = ['UP', 'DOWN', 'LEFT', 'RIGHT']

new_rows = []

combined = pd.read_csv("combined.csv")
print(combined)

for index, row in combined.iterrows():
    common_values = {col: row[col] for col in columns}

    for val in unique_values:
        row_data = common_values.copy()
        row_data['action_taken'] = val
        new_rows.append(row_data)

new_df = pd.DataFrame(new_rows)

action_encoder = OneHotEncoder(sparse=False)
actions = np.array(df['action_taken']).reshape(-1, 1)
action_encoder.fit(actions)

result_encoder = OneHotEncoder(sparse=False)
results = np.array(df['final_result']).reshape(-1, 1)
result_encoder.fit(results)

def one_hot_encode_actions(labels):
    return action_encoder.transform(np.array(labels).reshape(-1, 1))

def one_hot_encode_results(labels):
    return result_encoder.transform(np.array(labels).reshape(-1, 1))

def one_hot_to_label(one_hot_vector):
    return result_encoder.inverse_transform(one_hot_vector.reshape(1, -1))[0, 0]


output_df = pd.DataFrame(columns=['Unnamed: 0', 'bot_position_x', 'bot_position_y', 'actual_crew_pos_x',
       'actual_crew_pos_y', 'up_x', 'up_y', 'down_x', 'down_y', 'left_x',
       'left_y', 'right_x', 'right_y', 'up_alien_prob', 'up_crew_prob',
       'down_alien_prob', 'down_crew_prob', 'left_alien_prob',
       'left_crew_prob', 'right_alien_prob', 'right_crew_prob',
       'max_prob_cell_x', 'max_prob_cell_y', 'next_position_x',
       'next_position_y', 'up_rescued_percent', 'down_rescued_percent', 'left_rescued_percent', 'right_rescued_percent'])

columns = ['Unnamed: 0', 'bot_position_x', 'bot_position_y', 'actual_crew_pos_x',
           'actual_crew_pos_y', 'up_x', 'up_y', 'down_x', 'down_y', 'left_x',
           'left_y', 'right_x', 'right_y', 'up_alien_prob', 'up_crew_prob',
           'down_alien_prob', 'down_crew_prob', 'left_alien_prob',
           'left_crew_prob', 'right_alien_prob', 'right_crew_prob',
           'max_prob_cell_x', 'max_prob_cell_y', 'next_position_x',
           'next_position_y']

df = new_df
count_row=0
p1=0
p2=0
p3=0
p4=0
new_rows=[]

print(W)
print(df)

for index, row in df.iterrows():
    print(index)
    single_row_df = pd.DataFrame([row])
    processed_row_df = single_row_df.drop(columns=['action_taken', 'actual_crew_pos_x', 'actual_crew_pos_y', 'next_position_x', 'next_position_y', 'final_result'])
    action_encoded = one_hot_encode_actions(row['action_taken'])
    action_encoded_df = pd.DataFrame(action_encoded, index=[0])
    dropped_row = pd.concat([processed_row_df.reset_index(drop=True), action_encoded_df], axis=1)
    text_label_columns = ['action_taken']

    flattened_array = dropped_row.values.flatten()

    processed_input = flattened_array[:24]
    dropped_row=processed_input

    result = predict_percent_rescued_for_row(row, dropped_row, W, b, text_label_columns)
    if (count_row==0):
        p1=result
    elif (count_row==1):
        p2=result
    elif (count_row==2):
        p3=result
    elif (count_row==3):
        p4=result
    count_row+=1
    if (count_row==4):
        common_values = {col: row[col] for col in columns}
        row_data = common_values.copy()
        row_data['up_rescued_percent'] = p1
        row_data['down_rescued_percent'] = p2
        row_data['left_rescued_percent'] = p3
        row_data['right_rescued_percent'] = p4
        new_rows.append(row_data)
        count_row=0
    
Critic_df = pd.DataFrame(new_rows)
Critic_df.to_csv("Critic_df.csv")
print(Critic_df)