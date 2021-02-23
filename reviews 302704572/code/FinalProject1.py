
import numpy as np
import pandas as pd
import csv
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split 
from sklearn import metrics


#import os
#import warnings
import plotly.express as px

def getCategorical():
	return [
			'Profile Age',
			'Gender', 
			'Marital Status'
			]
def getList():
	return [
					'Churches',
					'Resorts',
					'beaches',
					'Parks',
					'museum',
					'Malls',
					'Zoo',
					'Restaurants',
					'Pubs/Bars',
					'Local Services',
					'Burger/Pizza shops',
					'Hotels/Other Lodgings',
					'Dance Clubs',
					'Swimming Pols',
					'Gyms',
					'Bakeries',
					'Beauty & Spas',
					'Cafes',
					'View Points',
					'Monuments',
					'Gardens'
				 ]
def change_columns_name(df):
	"""
	get dataframe and change name of columns according to list.
	"""
	new_names = [
					'Profile Age',
					'Gender', 
					'Marital Status',
					'Churches',
					'Resorts',
					'beaches',
					'Parks',
					'museum',
					'Malls',
					'Zoo',
					'Restaurants',
					'Pubs/Bars',
					'Local Services',
					'Burger/Pizza shops',
					'Hotels/Other Lodgings',
					'Dance Clubs',
					'Swimming Pols',
					'Gyms',
					'Bakeries',
					'Beauty & Spas',
					'Cafes',
					'View Points',
					'Monuments',
					'Gardens'
				 ]
	df.columns = new_names

def handle_Local_services_column(df):
	"""
	get dataframe check error values in 'Local Services rating average' column,
	fix errors and change values from object to float
	"""
	#check error values
	print(df['Local Services'].value_counts())

	#fix error value
	df['Local Services'][df['Local Services'] == '2\t2.'] = '2.2'

	#change object value to float
	df['Local Services'] = df['Local Services'].astype(float)

def categorical_to_numeric(df):
	"""
	change categorical values to numeric in 3 columns
	"""
	#check unique values
	for c in getCategorical():
		print(df[c].unique())

	#change categorical to numeric
	df['Profile Age'] = df["Profile Age"].replace({'<5': 0, '5-10': 1, '>10': 2})
	df['Gender'] = df['Gender'].replace({'male': 0, 'female': 1, '?': 2})
	df['Marital Status'] = df['Marital Status'].replace({'Single': 0, 'Married': 1})

def pairCorrelation(f1, f2, df):
		df['Profile Age'] = df["Profile Age"].replace({0:'<5', 1:'5-10', 2:'>10'})
		df['Gender'] = df['Gender'].replace({0:'male', 1:'female', 2:'?'})
		df['Marital Status'] = df['Marital Status'].replace({0:'Single', 1:'Married'})
		for f in getCategorical():
			sns.scatterplot(x=f1, y=f2, hue = f,data=df)
			plt.show()

def handle_Resorts(df):
	'''
	we can see there are 7 values > 5 --> outliners: remove it
	'''
	print(df['Resorts'][df['Resorts'] > 5].value_counts())
	df['Resorts'] = df['Resorts'].replace({5.99:2.32, 6.16:2.32, 5.90:2.32, 6.10:2.32, 6.03:2.32, 5.97:2.32, 6.14:2.32})

def parisCorrelations(df_03):
	pairCorrelation('Restaurants', 'Pubs/Bars', df_03)
	#pairCorrelation('Churches','Gardens', df_03)
	pairCorrelation('museum', 'Parks', df_03)
	pairCorrelation('Zoo','Restaurants', df_03)
	pairCorrelation('Zoo', 'Pubs/Bars', df_03)
	pairCorrelation('Hotels/Other Lodgings','Dance Clubs', df_03)
	pairCorrelation('Gyms','Beauty & Spas', df_03)	


def boxplots(df_03):
	sns.boxplot(x="museum", y="Profile Age", data=df_03)		
	plt.show()
	sns.boxplot(x="museum", y="Gender", data=df_03)		
	plt.show()
	sns.boxplot(x="museum", y="Marital Status", data=df_03)			
	plt.show()

def heatmap(df_03):
	plt.figure(figsize=(15,15))
	sns.heatmap(df_03.corr(), annot=True, cmap="coolwarm")
	plt.show()

def correlationCategoriesToMuseum(df_03):
	for f in getList():
		if f!="museum":
			#sns.scatterplot(x='museum', y=f, hue = 'Gender',data=df_03)
			df_03.plot.scatter(x='museum',y=f)
		plt.show()

def split_data(col, df):
	"""
	Splitting the data set into feature vector X and target variable y
	"""
	X = df.drop(col, axis=1)
	Y = df[col] # Target variables
	return X,Y

def GNB(df,x, y, h):
	'''
	Gaussian Naiv Bayes - get 2 features, target and data frame
	make prediction and show the results
	'''
	sns.scatterplot(x=x, y=y, hue=h,data=df)
	plt.show()
	## ##### Splitting the data set into feature vector X and target variable y
	X = df[[x,h]]
	y = df[y]
	Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=1)
	#from sklearn.naive_bayes import GaussianNB # 1. choose model class
	model = GaussianNB()                       # 2. instantiate model
	model.fit(Xtrain, ytrain)                  # 3. fit model to data
	y_model = model.predict(Xtest)             # 4. predict on new data (output is numpy array)

	ypred = pd.Series(y_model,name="prediction")
	predicted = pd.concat([Xtest.reset_index(),ytest.reset_index(),ypred],axis=1)
	
	#compare between prediction to target and ass column if correct or not
	predicted['correct']=np.nan
	predicted['correct'] = np.where((predicted['museum'] == predicted['prediction']), 1,0)
	print(predicted)

	print(x," ", h," accuracy: ", round(metrics.accuracy_score(ytest, y_model)*100,2),"%")
	#print number of correct predictions
	print(predicted.prediction.value_counts())
	sns.scatterplot(x=x, y=y, hue = 'correct',data=predicted)
	plt.show()

#List of metric for classiffication models
def metrics_classific(y,predicted,X):
    confusion_matrix1 = confusion_matrix(y, predicted)
    print(confusion_matrix)
    print(classification_report(y, predicted))
    print("Accuracy: %.2f%%" % (accuracy_score(y, predicted) * 100.0))

def bayes_plot(X,y,model="gnb",spread=30):
    df.dropna()
    colors = 'seismic'
    col1 = df.columns[0]
    col2 = df.columns[1]
    target = df.columns[2]
    sns.scatterplot(data=df, x=col1, y=col2,hue=target)
    plt.show()
    y = df[target]  # Target variable
    X = df.drop(target, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)  # 70% training and 30% test

    clf = GaussianNB()
    if (model != "gnb"):
        clf = DecisionTreeClassifier(max_depth=model)
    clf = clf.fit(X_train, y_train)
    
    # Train Classifer

    prob = len(clf.classes_) == 2

    # Predict the response for test dataset

    y_pred = clf.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(confusion_matrix)
    print(metrics.classification_report(y_test, y_pred))
    

    hueorder = clf.classes_
    def numify(val):
        return np.where(clf.classes_ == val)[0]

    Y = y.apply(numify)
    x_min, x_max = X.loc[:, col1].min() - 1, X.loc[:, col1].max() + 1
    y_min, y_max = X.loc[:, col2].min() - 1, X.loc[:, col2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2))

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    if prob:

        Z = Z[:,1]-Z[:,0]
    else:
        colors = "Set1"
        Z = np.argmax(Z, axis=1)


    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=colors, alpha=0.5)
    plt.colorbar()
    if not prob:
        plt.clim(0,len(clf.classes_)+3)
    sns.scatterplot(data=df[::spread], x=col1, y=col2, hue=target, hue_order=hueorder,palette=colors)
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    plt.show()
    
    aa  = df[::spread]
    return aa, spread
    
def func():

	# ####### INTRO ####### #
	print("# ####### INTRO ####### #")
	df_00 = pd.read_csv("reviews1.csv")
	print(df_00.info())
	# ####### INITIAL DATA ANALYSIS ####### #

	df_01 = df_00.copy()
	# change name of columns
	change_columns_name(df_01)

	# handling missing data
	print(df_00.isnull().sum())
	print(df_00[df_00['Gender']=='?'].shape)

	#delete all rows with missing data from Gender and Profile Age
	df_01 = df_01.dropna(subset=['Gender','Profile Age'])
	
	#we can see that 3 first column equal rows
	print(df_01.info())

	#handle Local Services column: 
	handle_Local_services_column(df_01)
	
	#show number of people that rated each category
	column_names = getList()
	counts = df_01[column_names[:]].astype(bool).sum(axis=0).sort_values()
	test = []
	for i in range(len(counts.index)):
		test.append(counts.index[i])
	fig = px.bar(counts, 
				 x=counts, 
				 y=test,
				 color=counts,
								 labels={
						 "total ratings": "this is x",
						 "categories": "this is y)"
					 },
					height = 800,
					title="Number of reviews under each category")
	fig.show()

	#fix missing numerical data to mean
	df_02 = df_01.copy()
	df_02= df_02.fillna(df_02.mean())

	#Local_services was change to float fron object
	print(df_02.info())
	
	#change categorical to numeric
	df_03 = df_02.copy()
	categorical_to_numeric(df_03)
	print(df_03.head())

	#Handle outliers
	print(df_03.describe().T)
	# we can see there are 7 values > 5 --> outliners: remove it 
	handle_Resorts(df_03)
	df_03.to_csv('resulting_data.csv', index=False)
	

	## ####### Explority Data Analysis ####### #
	print("## ####### Explority Data Analysis ####### #")
	#show average of each category 1
	#we can see gym has the min mean rating and malls has the max mean rating 
	#New_cols = getList()
	#AvgR = df_03[New_cols[:]].mean()
	#AvgR = AvgR.sort_values()
	#plt.figure(figsize=(10,7))
	#plt.barh(np.arange(len(New_cols[:])), AvgR.values, align='center')
	#plt.yticks(np.arange(len(New_cols[:])), AvgR.index)
	#plt.ylabel('Categories')
	#plt.xlabel('Average Rating')
	#plt.title('Average Rating for each Category')
	#plt.show()
	
	#show average of each category 2
	fig = px.box(df_03, y = getList())
	fig.show()

	heatmap(df_03)
	#boxplots musum to category columns
	boxplots(df_03)

	

	#correlationCategoriesToMuseum(df_03)

	##check correlation between each two features with correlation =0.5 with each categoerical features
	#parisCorrelations(df_03)

	 ####### ClassificationModel ####### #
	print("# ####### ClassificationModel ####### #")
	## Task 2 - 1
	print("## Task 2 - 1")
	GNB(df_03,'Parks','museum','Marital Status')
	GNB(df_03,'Parks','museum','Profile Age')
	GNB(df_03,'beaches','museum','Profile Age')
	GNB(df_03,'beaches','museum','Marital Status')

if __name__ == "__main__":
		func()