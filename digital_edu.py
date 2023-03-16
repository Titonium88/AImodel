#создай здесь свой индивидуальный проект!
import pandas as pd
df = pd.read_csv("train.csv")
#print(df.describe())

#Гипотеза: Люди не начавшие карьеру с большей вероятностью купят курс
#Очистка для построения модели
#id, bdate, followers_count, graduation, 'education_status', relation, education_form, life_main, people_main, city, last_seen, occupation_type, occupation_name
def langsCleaner(lang):
    return lang.count(';')+1

def careerCleaner(year):
    if year != 'False':
        return 1
    return 0

df['career_start'] = df['career_start'].apply(careerCleaner)

df['langs'] = df['langs'].apply(langsCleaner)
df['career_end'] = df['career_end'].apply(careerCleaner)

df.drop(['id' , 'has_photo' ,
        'bdate', 'followers_count',
        'graduation', 'education_status',
        'relation', 'education_form',
        'life_main', 'people_main',
        'city', 'last_seen',
        'occupation_type', 'occupation_name'], axis = 1 , inplace = True)

#model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

x = df.drop('result', axis = 1)
y = df['result']

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.25)

sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)

classifier = KNeighborsClassifier(n_neighbors = 3)

classifier.fit(train_x, train_y)

pred_y = classifier.predict(test_x)

print('Точность теста -',accuracy_score(test_y, pred_y)*100)
print('Confusion matrix')
confusion_matrix(test_y,pred_y)

