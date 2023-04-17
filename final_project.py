import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer

def main():
	dataset = pd.read_csv('data/Kaggle-data.csv', sep=',')
	X = dataset.drop(['ID', 'LoaderFlags', 'legitimate', 'NumberOfRvaAndSizes', 'Machine', 'SizeOfOptionalHeader', 'md5'], axis=1).values
	y = dataset['legitimate'].values

	# Splitting the dataset into the Training set and Test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 0)

	# Create imputer to replace missing values with the mean
	imp = SimpleImputer(missing_values=np.nan, strategy='mean')
	imp = imp.fit(X_train)
	X_train = imp.transform(X_train)
	# Feature Scaling
	#sc = StandardScaler()
	#X_train = sc.fit_transform(X_train)
	#X_test = sc.transform(X_test)


	#classifier = RandomForestClassifier(n_estimators=50, criterion = 'entropy', random_state = 0)
	#classifier = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
	#classifier = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
	#classifier = ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)
	#classifier = AdaBoostClassifier(n_estimators=100)
	#classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state =0)
	#classifier = KNeighborsClassifier(n_neighbors=5, p=2)
	classifier = MLPClassifier(random_state=1, max_iter=100)

	classifier.fit(X_train, y_train)

	# predict the test results
	X_test = imp.transform(X_test)
	y_pred = classifier.predict(X_test)

	ac = accuracy_score(y_pred, y_test)
	cm = confusion_matrix(y_test, y_pred)

	print("Accuracy: ", ac)
	print("Confusion_matrix: ", cm)

if __name__ == "__main__":
	main()



