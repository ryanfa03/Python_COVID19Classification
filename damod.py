import streamlit as st
st.title("Klasifikasi Tingkat Keparahan Covid-19 Berdasarkan Gejala yang Dialami Oleh Pasien Menggunakan Decision Tree Learning")

import numpy as np
import pandas as pd
import seaborn as sns;
import matplotlib.pyplot as plt
# import mglearn
# import graphviz
# import os
import warnings
warnings.filterwarnings('ignore')
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

st.header('Check semua gejala-gejala covid-19 yang dialami')

TotalSympton = 0
Symptoms = []
Fever = st.checkbox('Fever')
Tiredness = st.checkbox('Tiredness')
DryCough = st.checkbox('Dry-Cough')
Breathing = st.checkbox('Difficulty-in-Breathing')
SoreThroat = st.checkbox('Sore-Throat')
Pains = st.checkbox('Pains')
NasalCongestion = st.checkbox('Nasal-Congestion')
RunnyNose = st.checkbox('Runny-Nose')
Diarrhea = st.checkbox('Diarrhea')
NoneExperiencing = st.checkbox('None_Experiencing')
# Severity = st.selectbox('Severity',['None','Mild','Moderate','Severe'])
age = st.number_input('Age', 0,99)

if Fever:
    Fever = 1
    TotalSympton+=1
else:
    Fever = 0

if Tiredness:
    Tiredness = 1
    TotalSympton+=1
else:
    Tiredness = 0

if DryCough:
    DryCough = 1
    TotalSympton+=1
else:
    DryCough = 0

if Breathing:
    Breathing = 1
    TotalSympton+=1
else:
    Breathing = 0

if SoreThroat:
    SoreThroat = 1
    TotalSympton+=1
else:
    SoreThroat = 0

if Pains:
    Pains = 1
    TotalSympton+=1
else:
    Pains = 0

if NasalCongestion:
    NasalCongestion = 1
    TotalSympton+=1
else:
    NasalCongestion = 0

if RunnyNose:
    RunnyNose = 1
    TotalSympton+=1
else:
    RunnyNose = 0

if Diarrhea:
    Diarrhea = 1
    TotalSympton+=1
else:
    Diarrhea = 0

if NoneExperiencing:
    NoneExperiencing = 1
    
else:
    NoneExperiencing = 0


if age >= 0 and age <= 9:
    age_group = 1
elif age >= 10 and age <= 19:
    age_group = 2
elif age >= 20 and age <= 24:
    age_group = 3
elif age >= 25 and age <= 59:
    age_group = 4
else:
    age_group = 5

if st.button('Submit'):
    dataset = pd.read_csv("processedCovid9Symptoms.csv")
    dataset.drop(columns=dataset.columns[0], axis=1,  inplace=True)
    dataset['Severity'].fillna("None", inplace=True)

    Symptoms.append(Fever)
    Symptoms.append(Tiredness)
    Symptoms.append(DryCough)
    Symptoms.append(Breathing)
    Symptoms.append(SoreThroat)
    Symptoms.append(Pains)
    Symptoms.append(NasalCongestion)
    Symptoms.append(RunnyNose)
    Symptoms.append(Diarrhea)
    Symptoms.append(NoneExperiencing)
    Symptoms.append(age_group)
    Symptoms.append(TotalSympton)

    Severity='None'
    symp = {'Fever':[Fever],
        'Tiredness':[Tiredness],
        'Dry-Cough':[DryCough],
        'Difficulty-in-Breathing':[Breathing],
        'Sore-Throat':[SoreThroat],
        'Pains':[Pains],
        'Nasal-Congestion':[NasalCongestion],
        'Runny-Nose':[RunnyNose],
        'Diarrhea':[Diarrhea],
        'None_Experiencing':[NoneExperiencing],
        'Severity':[Severity],
        'Age':[age_group],
       }
    SymptomsRows = pd.DataFrame(symp)
    st.subheader('User Input')
    st.table(SymptomsRows)



    st.subheader('Base Dataset')
    st.table(dataset.tail(10))
    
    dataset = pd.concat([dataset, SymptomsRows], ignore_index=True)
    dataset.reset_index()
    st.subheader('Updated Dataset')
    st.table(dataset.tail(10))


    df = dataset.copy()
    df['Severity'].replace(['None','Mild','Moderate','Severe'],
                            [0, 1, 2, 3], inplace = True)

    st.subheader('Correlation Matrix')
    f,ax= plt.subplots(figsize=(10,10))
    sns.heatmap(df.corr(),annot=True)
    st.pyplot(f)

    st.subheader('Membagi Data Ke 4 Kelas')
    def count_percentage(columns, category):
        df = dataset[dataset[columns] == category]
        # print(df.shape)
        df = (df.shape[0]/dataset.shape[0])*100
        st.write(columns + ' ' + str(category) + ' Percentage: ', round(df), '%')

    count_percentage("Severity", "None")
    count_percentage("Severity", "Mild")
    count_percentage("Severity", "Moderate")
    count_percentage("Severity", "Severe")
    count_percentage("Sore-Throat", 1)



    symptoms = ['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat', 'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea']
    features = dataset[symptoms]
    temp = []
    for i in symptoms:
        temp.append(sum(features[i].values))
    temp_df = pd.DataFrame({"Symptons":symptoms, "Count":temp})
    fig, ax = plt.subplots()
    sns.barplot(data = temp_df, y="Symptons", x="Count", ax=ax)
    ax.set_xlim(0, dataset.shape[0])
    ax.set_title("Symptoms bar chart")
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy() 
        ax.annotate(f'{width/dataset.shape[0]:.0%}', 
                    (x + width*1.1, y + height*0.7), ha='center')
    st.subheader('Symptoms Bar Chart')
    st.pyplot(fig)



    def get_symptom_count(the_list):
        return sum(the_list.values)

    feats = dataset
    feats['Total_Sympton'] = feats[symptoms].apply(get_symptom_count, axis=1)

    fig, ax = plt.subplots()
    sns.countplot(data=feats, x='Total_Sympton', hue='Severity')
    plt.xlabel("Jumlah sympton yang diidap oleh pasien Covid-19")
    # plt.show()
    st.subheader('Symptoms Count Plot')
    st.pyplot(fig)



    drop = ['Severity']

    X = dataset
    X = X.drop(columns = drop)
    y = dataset['Severity']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify = y, random_state=42)

    param_dist = {"max_depth": [3, None],
             "min_samples_leaf": randint(1,9),
             "criterion": ["gini", "entropy"]}
    
    cross_val = StratifiedKFold(n_splits = 10)

    tree=DecisionTreeClassifier(random_state=0)

    tree_cv = RandomizedSearchCV(tree,param_dist, cv=cross_val)

    tree_cv.fit(x_train, y_train)

    st.subheader('Tuned Decision Tree Parameters')
    st.write(tree_cv.best_params_)
    st.write("Best score is {:.3f}".format(tree_cv.best_score_))

    y_pred = tree_cv.predict(x_test)

    st.subheader('Classification Report')
    st.write(classification_report(y_test, y_pred))
    

    tree = DecisionTreeClassifier(max_depth = 3, criterion="entropy", 
                              min_samples_leaf=2, random_state =0)
    tree.fit(x_train, y_train)

    st.subheader('Decision Tree Accuracy')
    st.write("Accuracy on training set: {:.3f}".format(tree.score(x_train, y_train)))
    st.write("Accuracy on test set: {:.3f}".format(tree.score(x_test, y_test)))

    from sklearn.tree import export_graphviz
    columns = X.columns
    

    st.subheader('Decision Tree Model')
    st.image("dtc.jpg")


    st.subheader('Feature Importance')
    # st.write(pd.DataFrame(tree.feature_importances_))
    def plot_feature_importances(model):
        n_features = X.shape[1]
        plt.barh(range(n_features), model.feature_importances_, align='center')
        plt.yticks(np.arange(n_features), columns)
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")
    plot_feature_importances(tree)

    fig, ax = plt.subplots()
    plot_feature_importances(tree)
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)


    st.subheader('Confusion Matrix')
    color = 'black'
    fig, ax = plt.subplots()
    matrix = plot_confusion_matrix(tree, x_test, y_test, cmap=plt.cm.Blues, ax=ax)
    matrix.ax_.set_title('Confusion Matrix', color=color)
    matrix.ax_.set_xlabel('Predicted Label', color=color)
    matrix.ax_.set_ylabel('True Label', color=color)
    matrix.ax_.xaxis.label.set_color(color)
    matrix.ax_.yaxis.label.set_color(color)
    matrix.ax_.tick_params(axis='x', colors=color)
    matrix.ax_.tick_params(axis='y', colors=color)
    st.pyplot(fig)
    
    

    st.header('Results')
    new_patient_symptoms = Symptoms
    severity = tree.predict([new_patient_symptoms])
    st.subheader(f"Berdasarkan Gejala-Gejala yang diinput maka anda memiliki tingat keparahan COVID-19 **{severity[0]}**")
else:
    st.write(" ")
