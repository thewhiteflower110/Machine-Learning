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
		
	