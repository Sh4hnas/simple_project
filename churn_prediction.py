import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

df= pd.read_csv("Telco_Customer_Churn.csv")
df.dropna(inplace=True)  
df['TotalCharges']= pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)  

le= LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    if col!= 'customerID':
        df[col]= le.fit_transform(df[col])

X= df.drop(['customerID', 'Churn'], axis=1)
y= df['Churn'] 

scaler= StandardScaler()
X_scaled= scaler.fit_transform(X)

X_train, X_test, y_train, y_test= train_test_split(X_scaled, y, test_size=0.2,random_state=42,stratify=y)

gb= GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)

y_pred= gb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

feature_importance= pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title("Feature Importance - Gradient Boosting")
plt.show()