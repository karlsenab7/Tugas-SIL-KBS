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


df = pandas.read_csv('../speedDating.csv', low_memory=False)

fields = df.field.str.lower().unique()
field_choices = []
for i in range(219):
    field_choices.append(i+1)
field_choices[218]

df.replace(fields,
           field_choices, inplace=True)

df.replace(['male', 'female'],
           [0, 1], inplace=True)

col_names = df.columns.drop(['match', 'id', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking',
                             'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'interests_correlate',
                             'like', 'guess_prob_liked', 'met', 'expected_happy_with_sd_people', 'expected_num_interested_in_me', 'expected_num_matches', 'attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'attractive_partner', 'sincere_partner',
                             'intelligence_partner', 'funny_partner', 'ambition_partner', 'shared_interests_partner', 'pref_o_attractive', 'pref_o_sincere',
                             'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests', 'attractive_o', 'sinsere_o', 'intelligence_o', 'funny_o', 'ambitous_o', 'decision', 'decision_o',
                             'shared_interests_o', 'attractive_important', 'sincere_important', 'intellicence_important', 'funny_important', 'ambtition_important',
                             'shared_interests_important'])


X_train, X_test, y_train, y_test = train_test_split(
    df[col_names], df['match'], test_size=0.3, random_state=100)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
r_predict = clf.predict(X_test)


dt_clf_gini = DecisionTreeClassifier(criterion="gini",
                                     random_state=100,
                                     max_depth=5,
                                     min_samples_leaf=5)

dt_clf_gini.fit(X_train, y_train)
y_pred_gini = dt_clf_gini.predict(X_test)

acc_gini = accuracy_score(y_test, y_pred_gini) * 100


filePath = sys.path[0] + "/model.pkl"
pickle.dump(dt_clf_gini, open(filePath, 'wb'))
print("saved to", filePath)
