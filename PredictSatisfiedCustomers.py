
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import xgboost as xgb
from sklearn import ensemble
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import roc_curve,roc_auc_score as auc
import matplotlib.pyplot as plt

#Import Data 
df = pd.read_csv(r'C:\Utkarsh\GIT\Python\PredictSatisfiedCustomers\Data\train.csv')
df_test = pd.read_csv(r'C:\Utkarsh\GIT\Python\PredictSatisfiedCustomers\Data\test.csv')

y  = df['TARGET']
df = df.drop('TARGET',axis=1)
df = df.drop('ID',axis=1)
df_test = df_test.drop('ID',axis=1)

#Dropping columns having least variance impact
sel2 = VarianceThreshold(threshold = .9)
np2 = sel2.fit_transform(df)
df = pd.DataFrame(np2)
np_test2 = sel2.transform(df_test)
df_prediction = pd.DataFrame(np_test2)

#Cross validation for removing over fitting
df_fit, df_eval, y_fit, y_eval= train_test_split( df, y, test_size=0.1, random_state=2 )

#First predictive model using XGboost 
xgboosting_model = xgb.XGBClassifier(missing=9999999999,max_depth = 5,n_estimators=100,
                                     learning_rate=0.1,nthread=4,subsample=.7)
xgboosting_model.fit(df_fit, y_fit)
predict_target = xgboosting_model.predict_proba(df_eval)[:,1]
validAUC = auc(y_eval, predict_target)
print("Accuracy with misssing value imputation"+validAUC)

#ROC curve and comparison with other models
names = ["etsc","abc","xgb","gbc"]
clfs = [
ensemble.ExtraTreesClassifier(n_estimators=100,max_depth=5),
ensemble.AdaBoostClassifier(n_estimators=100),
xgb.XGBClassifier(n_estimators=100, nthread=-1, max_depth = 5),
ensemble.GradientBoostingClassifier(n_estimators=100,max_depth=5)
]
plt.figure()
for name,clf in zip(names,clfs):

	clf.fit(df_fit, y_fit)
	predict_target = clf.predict_proba(df_eval)[:,1]
	print("Roc AUC:"+name, auc(y_eval, clf.predict_proba(df_eval)[:,1],average='macro'))
	fpr, tpr, thresholds = roc_curve(y_eval, predict_target)
	plt.plot(fpr, tpr, label=name)

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')

plt.show()          

