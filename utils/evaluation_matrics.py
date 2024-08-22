from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def print_validasi(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print("Akurasi: %0.3f" % accuracy)
    print("Presisi: %0.3f" % precision)
    print("Recall: %0.3f" % recall)
    print("F1 validasi: %0.3f" % f1)

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    plt.figure(figsize=(1, 1))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def print_score_uji(y_test, y_pred, gs):
    print(f'Akurasi model terbaik: {accuracy_score(y_test, y_pred):.3f}')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    print(f'Precision model terbaik: {precision:.3f}')
    print(f'Recall model terbaik: {recall:.3f}')
    f1_macro = f1_score(y_test, y_pred, average="macro")
    print(f'F1 pengujian model terbaik: {f1_macro:.3f}')
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # print(f'F1 validasi model terbaik: {gs.best_score_:.3f}')
    # print(f'Parameter terbaik: {gs.best_params_}')
    
    
def print_score_opt_uji(y_test, y_pred, gs):
    print(f'Akurasi model terbaik: {accuracy_score(y_test, y_pred):.3f}')
    print(f'Precision model terbaik: {precision_score(y_test, y_pred):.3f}')
    print(f'Recall model terbaik: {recall_score(y_test, y_pred):.3f}')
    f1_macro = f1_score(y_test, y_pred, average="macro")
    print(f'F1 pengujian model terbaik: {f1_macro:.3f}')
    print(f'F1 validasi model terbaik: {gs.best_value:.3f}')
    print(f'Parameter terbaik: {gs.best_params}')