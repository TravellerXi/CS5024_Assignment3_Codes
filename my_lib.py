import os
import pandas as pd
import cbor
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression
import time
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")



    
    
def get_instrument_name(code):
    instrument_map = {
        'REC': 'Recorder',
        'GUI': 'Guitar',
        'ACC': 'Accordion',
        'VL': 'Violin',
        'CL': 'Cello',
        'KEY': 'Keyboard',
        'FLU': 'Flute',
        'CON': 'Concertina',
    }
    return instrument_map.get(code, 'Unknown')

def load_cbor_file(filepath):
    with open(filepath, 'rb') as f:
        data = cbor.load(f)

    payload = data['payload']
    values = payload['values']
    df = pd.DataFrame(values, columns=['x', 'y', 'z'])
    filename = os.path.splitext(os.path.basename(filepath))[0]
    df['filename'] = filename
    df['instrument'] = df['filename'].str[3:6].apply(get_instrument_name)

    return df


def load_data_from_subfolders(main_folder):
    all_data = []

    for subdir, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith('.cbor'):
                filepath = os.path.join(subdir, file)
                #print(f'Loading data from {filepath}')
                df = load_cbor_file(filepath)
                all_data.append(df)

    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

def git_clone(PATH):
    current_dir = os.getcwd()
    if os.path.exists(os.path.join(current_dir,'CS5024_Assignment3_Instruments_Data')):
        return
    command = 'git clone ' + PATH
    os.system(command)

def create_segments(data, window_size=100, step_size=50):
    segments = []
    labels = []
    for _, group in data.groupby('filename'):
        instrument = group['instrument'].iloc[0]
        for i in range(0, len(group) - window_size + 1, step_size):
            segment = group[['x', 'y', 'z']].iloc[i:i + window_size].to_numpy()
            segments.append(segment)
            labels.append(instrument)
    return [np.array(segments), labels]

# Method: fetch_data
# Inputs: git_url:textd
# Output: data:Table
def fetch_data(git_url):
    git_clone(git_url)
    current_dir = os.getcwd()
    PATH = os.path.join(current_dir,'CS5024_Assignment3_Instruments_Data')
    data = load_data_from_subfolders(PATH)
    #print(data['instrument'].unique())
    return data


def create_segments(data, window_size=100, step_size=50):
    segments = []
    labels = []
    for _, group in data.groupby('filename'):
        instrument = group['instrument'].iloc[0]
        for i in range(0, len(group) - window_size + 1, step_size):
            segment = group[['x', 'y', 'z']].iloc[i:i + window_size].to_numpy()
            segments.append(segment)
            labels.append(instrument)
    return [np.array(segments), labels]



# Method: preprocess_data
# Inputs: data:Table
# Output: res:preprocess_data
def preprocess_data(data, window_size=100, step_size=50):
    # Drop rows with 'Unknown' instrument
    data = data[data['instrument'] != 'Unknown']
    
    X, y = create_segments(data, window_size, step_size)
    X = X.reshape(X.shape[0], -1)  # Flatten the segments for the classifier
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return [X, y_encoded, label_encoder]

# Method: train_xgb_classifier
# Inputs: input:preprocess_data
# Output:  res:trained_data
def train_xgb_classifier(preprocess_results):
    start_time = time.time()
    X, y, label_encoder = preprocess_results
    class RecordEvaluationCallback(xgb.callback.TrainingCallback):
        def __init__(self, results):
            self.results = results
        
        def after_iteration(self, model, epoch, evals_log):
            self.results["validation_0"] = {metric: evals_log["validation_0"][metric][-1] for metric in evals_log["validation_0"].keys()}
            self.results["validation_1"] = {metric: evals_log["validation_1"][metric][-1] for metric in evals_log["validation_1"].keys()}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    classifier = xgb.XGBClassifier(eval_metric=['mlogloss', 'merror'], early_stopping_rounds=10)
    
    eval_set = [(X_train, y_train), (X_test, y_test)]
    eval_result = {}
    classifier.fit(X_train, y_train, eval_set=eval_set, verbose=False, callbacks=[RecordEvaluationCallback(eval_result)])
    
    y_pred = classifier.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("--- %s minutes spent on training: ---" % (int(time.time() - start_time)/60))
    print('\n')
    return [ y_pred, y_test, score, eval_result, label_encoder]


# Method: train_ClassifierChain
# Inputs: input:preprocess_data
# Output:  res:trained_data
def train_ClassifierChain(preprocess_results):
    start_time = time.time()
    X, y, label_encoder = preprocess_results
    class RecordEvaluationCallback(xgb.callback.TrainingCallback):
        def __init__(self, results):
            self.results = results
        
        def after_iteration(self, model, epoch, evals_log):
            self.results["validation_0"] = {metric: evals_log["validation_0"][metric][-1] for metric in evals_log["validation_0"].keys()}
            self.results["validation_1"] = {metric: evals_log["validation_1"][metric][-1] for metric in evals_log["validation_1"].keys()}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    classifier = ClassifierChain(LogisticRegression())
    
    
    eval_result = {}
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    y_pred = y_pred.toarray()
    score = accuracy_score(y_test, y_pred)
    print("--- %s minutes spent on training: ---" % (int(time.time() - start_time)/60))
    print('\n')
    return [y_pred, y_test, score, eval_result, label_encoder]

# Method: train_BinaryRelevance_classifier
# Inputs: input:preprocess_data
# Output:  res:trained_data
def train_BinaryRelevance_classifier(preprocess_results):
    start_time = time.time()
    X, y, label_encoder = preprocess_results
    class RecordEvaluationCallback(xgb.callback.TrainingCallback):
        def __init__(self, results):
            self.results = results

        def after_iteration(self, model, epoch, evals_log):
            self.results["validation_0"] = {metric: evals_log["validation_0"][metric][-1] for metric in evals_log["validation_0"].keys()}
            self.results["validation_1"] = {metric: evals_log["validation_1"][metric][-1] for metric in evals_log["validation_1"].keys()}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier = BinaryRelevance(GaussianNB())

    eval_set = [(X_train, y_train), (X_test, y_test)]
    eval_result = {}
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    y_pred = y_pred.toarray()
    score = accuracy_score(y_test, y_pred)
    print("--- %s minutes spent on training: ---" % (int(time.time() - start_time)/60))
    print('\n')
    return [y_pred, y_test, score, eval_result, label_encoder]

def print_accuracy_score(score):
  print(f"Classifier accuracy: {score * 100:.2f}%")

def plot_confusion_matrix(y_true, y_pred, labels, figsize=(10, 8)):
    cm = confusion_matrix(y_true, y_pred)
    # normed_cm = (cm.T / cm.astype(np.float32).sum(axis=1)).T
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    
    plt.figure(figsize=figsize)
    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# Method: print_result
# Inputs:  trained_data:trained_data
def print_result(trained_data):
    y_pred, y_test, score, eval_result, label_encoder = trained_data
    print_accuracy_score(score)
    plot_confusion_matrix(y_test, y_pred, labels=label_encoder.classes_)
    

    






