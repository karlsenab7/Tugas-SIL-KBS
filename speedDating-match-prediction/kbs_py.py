import pandas
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

import os
import sys
import pickle

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


df = pandas.read_csv('speedDating.csv', low_memory=False)

fields = df.field.str.lower().unique()
field_choices = []
for i in range (219):
    field_choices.append(i+1)
field_choices[218]

df.replace(fields,
		field_choices, inplace=True)

df.replace(['male', 'female'],
            [0,1], inplace=True)

col_names = df.columns.drop(['match'])

X_train, X_test, y_train, y_test = train_test_split(
		df[col_names], df['match'], test_size = 0.3, random_state = 100)

clf_gini = DecisionTreeClassifier(criterion = "gini",
                                     random_state = 100,
                                     max_depth = 5,
                                     min_samples_leaf = 5)
 
clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)
 
acc_gini = accuracy_score(y_test, y_pred_gini) * 100
print("Accuracy Score: ", acc,"%")
print("F1 Score: ", f1,"%")
print ("Accuracy scrore using Gini: ",
             acc_gini,"%" )
print()

filePath = sys.path[0] + "speedDating-match-prediction\model.pkl"
pickle.dump(clf_gini, open(filePath, 'wb'))
print("saved to", filePath)
