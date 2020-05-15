class myeda:
	def __init__():
		import numpy as np
		import pandas as pd
		import matplotlib.pyplot as plt
		
	##GRAPH INFERENCE
	###Scatter Plot
	def scatter(x,y):
		plt.scatter(x,y)
		plt.show()
	
	def hist(df,no_of_bins):
		import matplotlib.pyplot as plt
		x = df #[value1, value2, value3,....]
		plt.hist(x, bins = no_of_bins)
		plt.show()

	##DATA CLEANING
	####Dealing with missing values
	def check_missing_value(df):
	# generate preview of entries with null values
	if len(df[df.isnull().any(axis=1)] != 0):
		print("Null values are present!!")
	
	def fill_missing_values_zeros(df):
		df.fill_na(0)
		return df
		
	def fill_missing_values_constant(df,constant):
		df.fill_na(constant)
		return df
	
	def fill_missing_values_mean(df,axis=1):
		mean_val=df.mean(axis=axis)
		df.fill_na(mean_val)
	
	def fill_missing_values_class(df,categorical_correlated_df):
		#5. Use the attribute mean for all samples belonging to the same class as the given tuple:
		#For example, if classifying customers according to credit risk, replace the missing value
		#with the average income value for customers in the same credit risk category as that
		#of the given tuple.
		index_names=df[df.isnull()].index
		d={}
		d.keys = pd.unique(df.to_numpy().ravel()) #get all categories
		for i in range[0,len(categorical_correlated_df)]:
			if(df[i]!=null):
				d[categorical_correlated_df[i]]=df[i]

		for key in d:
			d[key] = sum(d[key])/len(d[key])
		
		for i in range[0,len(df)]:
			if(df[i]==null):
				df[i]=d[categorical_correlated_df[i]] 
				
	###DEAL WITH NOISY DATA
	def check_smoothing(df):
		print("if the data is too scattered, it means the data needs smoothing")
		y=range(0,len(df))
		x=df
		scatter(x,y)
	
	###Deal with Outliers
	def check_outliers(df):
		print("If there are any values after the whiskers, then they are outliers")
		columns = df.select_dtypes(include=np.number).columns
		figure = plt.figure(figsize=(20, 10))
		figure.add_subplot(1, len(columns), 1)
		for index, col in enumerate(columns):
			if index > 0:
				figure.add_subplot(1, len(columns), index + 1)
			sns.boxplot(y=col, data=df, boxprops={'facecolor': 'None'})
		figure.tight_layout()
		plt.show()
	
	###Entity Identification problem
	def check_entity(df):
		if df.dtype !='category':
			print("Categorical Data is not allowed")
		else:
			print("Check the output list if, there are any issues in spelling of different classes in the dataframe")
			l=pd.unique(df.to_numpy().ravel())
			return l
		
	###Tuple Duplication problem
	def check_duplicates(df):
		#To check Duplicates
		if len(df[df.duplicated()]) > 0:
			print("No. of duplicated entries: ", len(df[df.duplicated()]))
			print(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)).head())
		else:
			print("No duplicated entries found")
		#If any duplicate values, correct them
		
	def solve_duplicates