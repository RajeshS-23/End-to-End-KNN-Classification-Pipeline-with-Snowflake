import pandas as pd
import snowflake.connector
conn=snowflake.connector.connect(
    user='username',
    password='password',
    account='account_id',
    database='db_name',
    schema='PUBLIC',
    warehouse='COMPUTE_WH'
)

pip install snowflake-connector-python

query='SELECT * FROM "table_name"'

import snowflake.connector
df=pd.read_sql(query,conn)
conn.close
print(df.head())

for col in df.columns:
  print(df[col].name,df[col].nunique())

df=df.drop(['C9','C10'],axis=1)

df.info()



for col in df.columns:
  print(df[col].name,df[col].value_counts())

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
num_cols=df.select_dtypes(include=['int','float']).columns
exc_col='C8'
num_cols=num_cols.drop(exc_col)
df[num_cols]=scaler.fit_transform(df[num_cols])

df.head(5)



df.head(5)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score,accuracy_score,roc_auc_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
x=df.drop(['C8'],axis=1)
y=df['C8']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=36,stratify=y)

print(y_test.value_counts())

from sklearn.metrics import classification_report, confusion_matrix

for k in range(3, 11):
    model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)

    print(f"\n===== k = {k} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision (macro):", precision_score(y_test, y_pred, average="macro"))
    print("Recall (macro):", recall_score(y_test, y_pred, average="macro"))
    print(
        "ROC AUC (macro OVR):",
        roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

import joblib
joblib.dump(model,'model.joblib')
joblib.dump(scaler,'scaling.joblib')

