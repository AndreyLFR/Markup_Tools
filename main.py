import pandas as pd
import numpy as np
import os
import warnings
import sys
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


warnings.filterwarnings('ignore')

# Task1. Деловая переписка часть корпоративной культуры любой компании.
# Гипотеза - Эмоции в переписке негативно сказываются на продуктивности взаимодействия коллег при решении задач
# Так как деловая переписка конфиденциальна для обучения взят открытый датасет c email компании The Enron


def get_sentiment(message):
    message_text = message.split('\n')[-1]
    return TextBlob(message_text).sentiment.polarity


def get_label(score):
    if score < 0:
        return 'neg'
    else:
        return 'pos'


def train_model(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['message'])
    y = df['target']
    model = LogisticRegression()
    model.fit(X, y)
    return model, vectorizer


df = pd.read_csv('emails.csv')
#создаю столбцы
df['target'] = np.nan
df['sentiment'] = np.nan
df = shuffle(df)
#неразмеченный датасет делю на 2 датасета. Первый размечу вручную и правилами. Второй размечу моделью
labeled_df, unlab_df = train_test_split(df, train_size=0.3, random_state=42)
# неразмеченную выборку делю на тестовую, которую заполню вручную и выборку, которую заполню моделью
test_df = unlab_df[:100]
unlabeled_df = unlab_df[100:]
#датасет делю на ручную и автоматическую разбивку
labeled_df_manual = labeled_df[100:]
labeled_df_rule = labeled_df[:100]
#разметка по правилам
labeled_df_rule['sentiment'] = labeled_df_rule['message'].apply(get_sentiment)
labeled_df_rule['target'] = labeled_df_rule['sentiment'].apply(get_label)
labeled_df_rule['target'] = labeled_df_rule['target'].replace(['neg', 'pos'], [0, 1])
labeled_df_rule['target'] = pd.to_numeric(labeled_df_rule['target'])
#разметка вручную
if os.path.exists('labeled_manual.csv') and os.path.exists('test.csv'):
    labeled_df_manual = pd.read_csv('labeled_manual.csv')
    test_df = pd.read_csv('test.csv')
else:
    labeled_df_manual.to_csv('labeled_manual.csv')
    test_df.to_csv('test.csv')
    print('После ручной разметки запусти повторно')
    sys.exit(1)
#объединяю размеченные датасеты
labeled_df = pd.concat([labeled_df_rule, labeled_df_manual])
# смотрю распределение и выравниванию по классам
labeled_df['target'].hist()
plt.show()
train_1 = labeled_df[labeled_df['target'] == 1]
train_0 = labeled_df[labeled_df['target'] == 0]
print(f'до ребалансировки количество объектов 0 класса {train_0.shape[0]}, 1 класса {train_1.shape[0]}')
train_1 = train_1.sample(train_0.shape[0], random_state=0)
train_bal = pd.concat([train_1, train_0])
print(f'после ребалансировки количество объектов 0 класса {train_0.shape[0]}, 1 класса {train_1.shape[0]}')
# заполняю неразмеченный датасет
model, vectorizer = train_model(df=train_bal)
X_unlabeled = vectorizer.transform(unlabeled_df['message'])
y_unlabeled = model.predict(X_unlabeled)
unlabeled_df['target'] = y_unlabeled
# объединяю полностью размеченный датасет
labeled_new_df = pd.concat([labeled_df, unlabeled_df])
# оцениваем модель на тестовой выборке, размеченной вручную
test_df = pd.read_csv('test.csv')
X_test = vectorizer.transform(test_df['message'])
y_predict = model.predict(X_test)
f1 = f1_score(test_df['target'], y_predict)
print(f'Модель демонстирует f1 score {f1}. '
      f'f1 score аккумулирует в себе значения точности модели и recall.'
      f'Модель на {round(f1*100, 2)}% верно разметила датасет')