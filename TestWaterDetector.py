
# Approach 2

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# df = pd.read_csv('water_potability.csv')
# df.head()
#
# df.shape
#
# df.isnull().sum()
#
# df.info()
#
# df.describe()
#
# df.fillna(df.mean(), inplace=True)
# df.isnull().sum()
#
# df.Potability.value_counts()
#
# sns.countplot(df['Potability'])
# plt.show()
#
# sns.distplot(df['ph'])
# plt.show()
#
# df.hist(figsize=(14,14))
# plt.show()
#
# plt.figure(figsize=(13,8))
# sns.heatmap(df.corr(),annot=True,cmap='terrain')
# plt.show()
#
# df.boxplot(figsize=(14,7))
#
# X = df.drop('Potability',axis=1)
# Y= df['Potability']
#
# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=101,shuffle=True)
#
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
# dt=DecisionTreeClassifier(criterion= 'gini', min_samples_split= 10, splitter= 'best')
# dt.fit(X_train,Y_train)
#
# prediction=dt.predict(X_test)
# print(f"Accuracy Score = {accuracy_score(Y_test,prediction)*100}")
# print(f"Confusion Matrix =\n {confusion_matrix(Y_test,prediction)}")
# print(f"Classification Report =\n {classification_report(Y_test,prediction)}")
#
# res = dt.predict([[5.735724, 158.318741,25363.016594,7.728601,377.543291,568.304671,13.626624,75.952337,4.732954]])[0]
# print(res)
#
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.model_selection import GridSearchCV
#
# # define models and parameters
# model = DecisionTreeClassifier()
# criterion = ["gini", "entropy"]
# splitter = ["best", "random"]
# min_samples_split = [2,4,6,8,10,12,14]
#
# # define grid search
# grid = dict(splitter=splitter, criterion=criterion, min_samples_split=min_samples_split)
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# grid_search_dt = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
# grid_search_dt.fit(X_train, Y_train)
#
# print(f"Best: {grid_search_dt.best_score_:.3f} using {grid_search_dt.best_params_}")
# means = grid_search_dt.cv_results_['mean_test_score']
# stds = grid_search_dt.cv_results_['std_test_score']
# params = grid_search_dt.cv_results_['params']
#
# for mean, stdev, param in zip(means, stds, params):
#     print(f"{mean:.3f} ({stdev:.3f}) with: {param}")
#
# print("Training Score:", grid_search_dt.score(X_train, Y_train) * 100)
# print("Testing Score:", grid_search_dt.score(X_test, Y_test) * 100)


# Approach 1

import cv2
import numpy as np
from keras.models import model_from_json

original = cv2.imread("pureWater2.jpg")
image_to_compare = cv2.imread("Satellite-image-of-oil-pollution.png")

original = cv2.resize(original, (1000, 650))
image_to_compare = cv2.resize(image_to_compare, (1000, 650))

# find the key points of both images and find their similarities
orb = cv2.ORB_create()
kp_1, desc_1 = orb.detectAndCompute(original, None)
kp_2, desc_2 = orb.detectAndCompute(image_to_compare, None)

# find match between key points
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = matcher.knnMatch(desc_1, desc_2, k=2)

# find the good key points of all
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append([m])

final_image = cv2.drawMatchesKnn(original, kp_1, image_to_compare, kp_2, good, None)
final_image = cv2.resize(final_image, (1000, 650))

if original.shape == image_to_compare.shape:
    difference = cv2.subtract(original, image_to_compare)
    b, g, r = cv2.split(difference)

    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        print("Water is Drinkable")

    elif len(good) >= 500:
        print("Water is Drinkable")

    else:
        print("Water is Not Drinkable")

cv2.imshow("matches", final_image)

# print(len(good))
# cv2.imshow("original ", original)
# cv2.imshow("image_to_compare ", image_to_compare)
# cv2.imshow("difference", difference)

cv2.waitKey(0)
cv2.destroyAllWindows()
