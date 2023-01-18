# Importar Bibliotecas

import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# Versão da biblioteca
import sklearn
versão = sklearn.__version__
print(versão)


# Visualizar arquivo
credit_card = pd.read_csv('/Users/reinaldoblack/Documents/documentos/github/streamlit/CreditCardFraud/creditcard.csv')
credit_card

# Dados nulos
credit_card.isnull().sum()

# Valores único

credit_card_columns = credit_card.columns

for values in credit_card_columns:
    print(len(credit_card[values].unique()))

# Visualizar as 10 primeiras letras

credit_card.head(10)

# Visualizar as médias e quartis

credit_card.describe()

# Converter coluna time

credit_card['datetime'] = pd.to_datetime(credit_card['Time'], unit='s')
credit_card['Time'] = credit_card['datetime'].dt.strftime('%H:%M:%S')
credit_card['datetime'] = credit_card['datetime'].dt.strftime('%H:%M:%S')

credit_card.drop(columns=['Time'], inplace=True)
credit_card.rename(columns={'datetime': 'Time'}, inplace=True)
credit_card

# Fraud vs no Fraud

print('No Fraud! :', round(credit_card['Class'].value_counts()[0] / len(credit_card) * 100,2 ))
print('Fraud! :', round(credit_card['Class'].value_counts()[1] / len(credit_card) * 100,2))

# Regressão linear
logistic_regressao = LogisticRegression()
X = credit_card.drop(columns=['Class', 'Time'])
y = credit_card['Class']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=2, test_size=0.3)

logistic_regressao.fit(X_train, y_train)

y_pred = logistic_regressao.predict_proba(X_test)
y_pred = y_pred[:,1]

a, b,_ = roc_curve(y_test, y_pred)
plt.plot(a,b)







