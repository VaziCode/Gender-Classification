
import plotly.express as px

import pandas as pd
import numpy as np
import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier


ds = pd.read_csv("../Transformed Data Set - Sheet1.csv")
ds.info()
X = ds.drop("Gender" , axis =1)
y= ds.Gender

X.head()

X.shape

print("------------------------------------------------")


print("------------------------------------------------")
print("Data initialization check")

print(X.isnull().sum())

print("------------------------------------------------")


object_cols = [col for col in X.columns if X[col].dtype==object]
print("OBJECT COLUMNS ARE" , object_cols)

num_cols = list(set(X.columns) - set(object_cols))
print("NUMERICAL COLUMNS ARE" , num_cols)



ds[ds["Favorite Color"] == "Cool"]["Gender"].value_counts(normalize=True)



ds[ds["Favorite Color"] == "Warm"]["Gender"].value_counts(normalize=True)



ds[ds["Favorite Color"] == "Neutral"]["Gender"].value_counts(normalize=True)



fig = px.histogram(data_frame=ds, x="Favorite Color", color="Gender", width=400, height=400)
fig.show()



fig = px.histogram(data_frame=ds, x="Favorite Music Genre", color="Gender", width=400, height=400)
fig.show()



fig = px.histogram(data_frame=ds, x="Favorite Beverage", color="Gender", width=400, height=400)
fig.show()


ds["Gender"] = ds["Gender"].map(lambda x: 0 if x=="F" else 1)

round(ds.groupby("Favorite Soft Drink")["Gender"].mean(), 2).sort_values(ascending=False)


fig = px.histogram(data_frame=ds, x="Favorite Soft Drink", color="Gender", width=400, height=400)
fig.show()


round(ds.groupby("Favorite Color")["Gender"].mean(), 2).sort_values(ascending=False)
# describing percentage vount in different values of feature


encoder = LabelEncoder()
project_data = ds.copy()
project_data['Favorite_Color_Transformed'] = encoder.fit_transform(ds['Favorite Color'])
project_data['Favorite_Music_Genre_Transformed'] = encoder.fit_transform(ds['Favorite Music Genre'])
project_data['Favorite_Beverage_Transformed'] = encoder.fit_transform(ds['Favorite Beverage'])
project_data['Favorite_Soft_Drink_Transformed'] = encoder.fit_transform(ds['Favorite Soft Drink'])
project_data['Gender_Transformed'] = encoder.fit_transform(ds['Gender'])


project_data.head()


training_data = project_data.drop(["Favorite Color" , "Favorite Music Genre", "Favorite Beverage", "Favorite Soft Drink" , "Gender"], axis= 1  )
training_data.head()
project_data.head()


corr = project_data.corr()
sns.heatmap(corr)


mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(170, 45, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


X = project_data.drop(['Favorite Color','Favorite Music Genre','Favorite Beverage','Favorite Soft Drink','Gender','Gender_Transformed'], axis=1)
Y = project_data['Gender']


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=101)

print("------------------------------------------------")
print("""Model #1 - Logistic Regression
-------------------------------------------------""")
logisticRegression_model = LogisticRegression()
logisticRegression_model.fit(X_train,Y_train)
predict = logisticRegression_model.predict(X_test)

print(classification_report(Y_test,predict))
print("------------------------------------------------")

print("""Model #2 - Decision Tree Classifier
-------------------------------------------------""")
decisionTreeClassifier_model = DecisionTreeClassifier()
decisionTreeClassifier_model.fit(X_train,Y_train)
predict = decisionTreeClassifier_model.predict(X_test)

print(classification_report(Y_test,predict))
print("------------------------------------------------")


from prettytable import PrettyTable

x = PrettyTable(["Model", "Avg Accuracy"])
x.add_row(["LogisticRegression","55.0"])
x.add_row(["DecisionTreeClassifier","55.0"])
print(x)

final_model = GradientBoostingClassifier()
final_model.fit(X,Y)
predict = final_model.predict(X)