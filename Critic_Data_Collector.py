import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

caught_df=pd.read_csv("Balanced_Actor_caught.csv")
rescued_df=pd.read_csv("Balanced_Actor_rescued.csv")


caught_combined = pd.concat([caught_df], ignore_index=True)


rescued_combined = pd.concat([rescued_df], ignore_index=True)

if len(rescued_combined) > len(caught_combined):
    rescued_combined = rescued_combined.sample(len(caught_combined))

df = pd.concat([rescued_df, caught_df])
df = df.drop(columns=['Unnamed: 0'])
df_original=df

print("########################################################################")
print(df)
print("##################################################################")

columns = ['bot_position_x', 'bot_position_y', 'actual_crew_pos_x',
           'actual_crew_pos_y', 'up_x', 'up_y', 'down_x', 'down_y', 'left_x',
           'left_y', 'right_x', 'right_y', 'up_alien_prob', 'up_crew_prob',
           'down_alien_prob', 'down_crew_prob', 'left_alien_prob',
           'left_crew_prob', 'right_alien_prob', 'right_crew_prob',
           'max_prob_cell_x', 'max_prob_cell_y', 'next_position_x',
           'next_position_y', 'final_result']

unique_values = ['UP', 'DOWN', 'LEFT', 'RIGHT']

new_rows = []

combined = pd.read_csv("combined.csv")

for index, row in combined.iterrows():
    common_values = {col: row[col] for col in columns}

    for val in unique_values:
        row_data = common_values.copy()
        row_data['action_taken'] = val
        new_rows.append(row_data)

new_df = pd.DataFrame(new_rows)
print("##################################################################")
print(new_df)
print("##################################################################")

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

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x > 0) * 1

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9))

def forward_pass(x, W, b):
    out = [x]
    in_hidden = np.dot(W[1], out[0]) + b[1]
    out_hidden = relu(in_hidden)
    out.append(out_hidden)
    in_output = np.dot(W[2], out[1]) + b[2]
    out_output = softmax(in_output)
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

def train(X_df, Y_encoded, W, b, epochs, learning_rate):
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X_df)):
            sample = X_df.iloc[i]
            y_true_encoded = Y_encoded[i]
            out = forward_pass(sample.values, W, b)
            y_pred = out[-1]
            loss = cross_entropy_loss(y_true_encoded, y_pred)
            total_loss += loss
            grad_W, grad_b = backward_pass(out, W, b, y_true_encoded)
            W, b = update_parameters(W, b, grad_W, grad_b, learning_rate)
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(X_df)}")
    return W, b

def test(X_df, Y_encoded, W, b):
    total_loss = 0
    correct_predictions = 0
    all_true_labels = []
    all_predicted_labels = []
    for i in range(len(X_df)):
        sample = X_df.iloc[i]
        y_true_encoded = Y_encoded[i]
        out = forward_pass(sample.values, W, b)
        y_pred = out[-1]
        loss = cross_entropy_loss(y_true_encoded, y_pred)
        total_loss += loss

        predicted_label = one_hot_to_label(y_pred)
        true_label = one_hot_to_label(y_true_encoded)

        all_true_labels.append(true_label)
        all_predicted_labels.append(predicted_label)

        if predicted_label == true_label:
            correct_predictions += 1

    average_loss = total_loss / len(X_df)
    accuracy = correct_predictions / len(X_df)
    return average_loss, accuracy, all_true_labels, all_predicted_labels

action_encoded = one_hot_encode_actions(df['action_taken'])
print(df)
df.reset_index(drop=True, inplace=True)
action_encoded_df = pd.DataFrame(action_encoded).reset_index(drop=True)
X_df = pd.concat([df.drop(columns=['action_taken', 'actual_crew_pos_x', 'actual_crew_pos_y', 'next_position_x', 'next_position_y', 'final_result']), action_encoded_df], axis=1)
print(X_df)
Y = df['final_result']
print(Y)

Y_encoded = one_hot_encode_results(Y)


X_train, X_test, Y_train_encoded, Y_test_encoded = train_test_split(X_df, Y_encoded, test_size=0.1, random_state=0)

input_size = X_train.shape[1]
hidden_layer_size = 100
output_size = Y_train_encoded.shape[1] 

W = [None, np.random.randn(hidden_layer_size, input_size), np.random.randn(output_size, hidden_layer_size)]
b = [None, np.random.randn(hidden_layer_size), np.random.randn(output_size)]

epochs = 100
learning_rate = 0.001

W, b = train(X_train, Y_train_encoded, W, b, epochs, learning_rate)

test_loss, test_accuracy, true_labels, predicted_labels = test(X_test, Y_test_encoded, W, b)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
W_df = pd.DataFrame(W)
b_df = pd.DataFrame(b)
W_df.to_csv("Critic_W_df.csv")
b_df.to_csv("Critic_b_df.csv")

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

cm = confusion_matrix(true_labels, predicted_labels, labels=result_encoder.categories_[0])
precision, recall = calculate_precision_recall(cm)

for i, label in enumerate(result_encoder.categories_[0]):
    print(f"Precision for {label}: {precision[i]:.2f}")
    print(f"Recall for {label}: {recall[i]:.2f}")

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

def forward_pass1(x, W, b):
    out = [x]
    in_hidden = np.dot(W[1], out[0]) + b[1]
    out_hidden = relu(in_hidden)
    out.append(out_hidden)
    in_output = np.dot(W[2], out[1]) + b[2]
    out_output = softmax(in_output)
    out.append(out_output)
    return out

def predict_percent_rescued_for_row(sample, sample_dropped, W, b, text_label_columns):
    
    action=sample['action_taken']
    x_processed = sample_dropped
    out = forward_pass1(x_processed, W, b)
    y_pred = out[-1]
    return y_pred[1]




output_df = pd.DataFrame(columns=['bot_position_x', 'bot_position_y', 'actual_crew_pos_x',
       'actual_crew_pos_y', 'up_x', 'up_y', 'down_x', 'down_y', 'left_x',
       'left_y', 'right_x', 'right_y', 'up_alien_prob', 'up_crew_prob',
       'down_alien_prob', 'down_crew_prob', 'left_alien_prob',
       'left_crew_prob', 'right_alien_prob', 'right_crew_prob',
       'max_prob_cell_x', 'max_prob_cell_y', 'next_position_x',
       'next_position_y', 'up_rescued_percent', 'down_rescued_percent', 'left_rescued_percent', 'right_rescued_percent'])

columns = ['bot_position_x', 'bot_position_y', 'actual_crew_pos_x',
           'actual_crew_pos_y', 'up_x', 'up_y', 'down_x', 'down_y', 'left_x',
           'left_y', 'right_x', 'right_y', 'up_alien_prob', 'up_crew_prob',
           'down_alien_prob', 'down_crew_prob', 'left_alien_prob',
           'left_crew_prob', 'right_alien_prob', 'right_crew_prob',
           'max_prob_cell_x', 'max_prob_cell_y', 'next_position_x',
           'next_position_y']

df=new_df
count_row=0
p1=0
p2=0
p3=0
p4=0
new_rows=[]

for index, row in df.iterrows():
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