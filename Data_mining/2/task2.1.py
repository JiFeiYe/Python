# Importing the basic libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
# import pandas_profiling as pdb
import os

plt.style.use("seaborn-darkgrid")

# Loading the data.
os.listdir("D:\\CODE\\Visual_Studio_Code_Python\\Data_mining\\2")

# Loading the training and testing data.
# we will check how good or bad the data is, that we already have.
train_data = pd.read_csv("D:\\CODE\\Visual_Studio_Code_Python\\Data_mining\\2\\train.csv")
test_data = pd.read_csv("D:\\CODE\\Visual_Studio_Code_Python\\Data_mining\\2\\test.csv")

# checking the training and testing data.
print(train_data.shape)
print(test_data.shape)

# checking the sample data for training data.
train_data.head(20)

# for each columns, let's quickly look into the distinct values as well.
# this is one of the ways in which we can understand any gaps in the data model.
for each_col in train_data.columns:
    print(each_col)
    print(train_data[each_col].value_counts())
    print("")

# checking basic statistics of training data.
train_data.describe()

# Checking the information about the data.
# this will tell us how many and what type of columns/rows we have in the data.
train_data.info()

# Creating a list of columns which are numeric.
numeric_cols = list(train_data.select_dtypes(exclude="object").columns)
numeric_cols

# creating a list of columns which are categorical or contains string values.
string_cols = list(train_data.select_dtypes(include="object").columns)
string_cols

# For each numerical column, let's check the data distribution.
# for this we are using seaborn library and try performing pairplotting
sns.pairplot(train_data[numeric_cols])
plt.show()

# Clearly we do not need the Passenger ID in this analysis, which we can drop at this stage.
train_data.drop("PassengerId", axis=1, inplace=True)
test_data.drop("PassengerId", axis=1, inplace=True)

# Creating again, after dropping the passenger ID column, the list of columns which are numeric.
numeric_cols = list(train_data.select_dtypes(exclude="object").columns)
numeric_cols

# # using pandas profiler once to look into all the data columns.
# train_data_profile = pdb.ProfileReport(train_data)
# train_data_profile

# it seems we have some missing values in certain columns.
# let's look into them again .
train_data.isna().sum()

# dropping the Cabin column.
train_data.drop("Cabin", axis=1, inplace=True)
test_data.drop("Cabin", axis=1, inplace=True)

## Checking the name column.
train_data["Name"].value_counts().head(20)

# Let's check the ticket column once.
train_data["Ticket"].value_counts().head(20)

# dropping the ticket column since we could not see any meaningful or logical patterns from the data.
train_data.drop("Ticket", axis=1, inplace=True)
test_data.drop("Ticket", axis=1, inplace=True)

# Re-checking the dataframe information.
train_data.info()

# What about the testng data?
# does it contain missing values as well ?
test_data.info()

# filling the missing data of Age in training and testing data.
for i in train_data.columns[train_data.isnull().any(axis=0)]:
    print(i)
    train_data[i].fillna(train_data[i].mode(), inplace=True)

## For the training and testing data, where Age is missing, we can perform imputation using mode
train_data["Age"] = train_data["Age"].fillna(
    (train_data["Age"].mode().astype("float64"))
)
test_data["Age"] = test_data["Age"].fillna((test_data["Age"].mode().astype("float64")))

# rechecking the training data.
train_data.info()

# Looks like the earlier imputation method did not work, hence trying other options.
from sklearn.impute import SimpleImputer as SI

imr = SI(missing_values=np.nan, strategy="median")
imr = imr.fit(train_data[["Age"]])
train_data[["Age"]] = imr.transform(train_data[["Age"]])

# Checking Age column now.
train_data["Age"].isna().sum()
train_data.info()

# Performing similar operation for Embarked column.
# since it is a categorical column, the 'strategy' will be different.
imr = SI(missing_values=np.nan, strategy="most_frequent")
imr = imr.fit(train_data[["Embarked"]])
train_data[["Embarked"]] = imr.transform(train_data[["Embarked"]])

# rechecking the dataframe information once more.
train_data.info()

## Let's repeat the above steps for the Testing data again.

# for Age column
imr = SI(missing_values=np.nan, strategy="median")
imr = imr.fit(test_data[["Age"]])
test_data[["Age"]] = imr.transform(test_data[["Age"]])

# for Embarked column
imr = SI(missing_values=np.nan, strategy="most_frequent")
imr = imr.fit(test_data[["Embarked"]])
test_data[["Embarked"]] = imr.transform(test_data[["Embarked"]])

# Checking the Testing data.
test_data.info()

## In the testing data we have Fare data missing.
## We'll treat it similarly.

imr = SI(missing_values=np.nan, strategy="median")
imr = imr.fit(test_data[["Fare"]])
test_data[["Fare"]] = imr.transform(test_data[["Fare"]])

# Rechecking the Testing data.
test_data.info()

# Extracting the Title information from Name.
train_data["Title"] = train_data["Name"].str.extract("([A-Za-z]+)\.", expand=False)
test_data["Title"] = test_data["Name"].str.extract("([A-Za-z]+)\.", expand=False)

# Check the counts of each titles
title_counts = train_data.Title.value_counts()
title_counts.to_frame().T

title_counts = test_data.Title.value_counts()
title_counts.to_frame().T

# check the sex of the passengers with different titles in the full dataset

# creating empty sets
female_titles = set()
male_titles = set()

# loop through data of Title column
# add values to the 'empty' sets
for t in train_data.Title.unique():
    if ((train_data.Title == t) & (train_data.Sex == "female")).any():
        female_titles.add(t)
    if ((train_data.Title == t) & (train_data.Sex == "male")).any():
        male_titles.add(t)

# There will be cases of mixture of titles applicable to both males and females.
mix_titles = female_titles & male_titles

# Finally creating the sets which only belongs to males and females
female_only_titles = female_titles - mix_titles
male_only_titles = male_titles - mix_titles

# Printing the findings.
print("mix_titles:", mix_titles)
print("female_only_titles", female_only_titles)
print("male_only_titles", male_only_titles)

# repeating above steps for Test Data too.

# creating empty sets
female_titles = set()
male_titles = set()

# loop through data of Title column
# add values to the 'empty' sets
for t in test_data.Title.unique():
    if ((test_data.Title == t) & (test_data.Sex == "female")).any():
        female_titles.add(t)
    if ((test_data.Title == t) & (test_data.Sex == "male")).any():
        male_titles.add(t)

# There will be cases of mixture of titles applicable to both males and females.
mix_titles = female_titles & male_titles

# Finally creating the sets which only belongs to males and females
female_only_titles = female_titles - mix_titles
male_only_titles = male_titles - mix_titles

# Printing the findings.
print("mix_titles:", mix_titles)
print("female_only_titles", female_only_titles)
print("male_only_titles", male_only_titles)

## Visualizing the Title information and Age
plt.figure(figsize=(20, 10))
sns.boxplot(x="Title", y="Age", data=train_data)
plt.show()

# change the title from 'Dr' to 'Dr (female)' for the female
train_data.loc[
    (train_data.Title == "Dr") & (train_data.Sex == "female"), "Title"
] = "Dr (female)"
test_data.loc[
    (test_data.Title == "Dr") & (test_data.Sex == "female"), "Title"
] = "Dr (female)"

# creating the title groupings
title_groups = {
    "Male adult": [
        "Mr",
        "Don",
        "Rev",
        "Dr",
        "Sir",
        "Major",
        "Col",
        "Capt",
        "Countess",
        "Jonkheer",
    ],
    "Boy": ["Master"],
    "Miss": ["Miss"],
    "Other female": ["Mrs", "Dona", "Mme", "Mlle", "Ms", "Lady", "Dr (female)"],
}

# adding the new column in the training data for title groups.
train_data["Title_group"] = train_data["Title"]
for k in title_groups:
    train_data["Title_group"].replace(title_groups[k], k, inplace=True)

# repeating the same for testing data.

test_data["Title_group"] = test_data["Title"]
for k in title_groups:
    test_data["Title_group"].replace(title_groups[k], k, inplace=True)

# checking the dataframe information
train_data.info()

# CHecking the value counts for title group.
train_data.Title_group.value_counts()

# CHecking the test data too.
test_data.Title_group.value_counts()

# We can now drop the Name column from the Training and Testing data.
train_data.drop("Name", axis=1, inplace=True)
test_data.drop("Name", axis=1, inplace=True)

# Checking the training data.
print(train_data.info())

# Importing the Label Encoder library
from sklearn.preprocessing import LabelEncoder

# Creating an object of Label Encoder
le = LabelEncoder()

# creating list of columns from training and testing data which are object types.
str_cols_train_data = train_data.select_dtypes(include="object")
str_cols_test_data = test_data.select_dtypes(include="object")

# looping through each columns and label encoding them for both Training and Testing data.

for each_col in str_cols_train_data:
    train_data[each_col] = le.fit_transform(train_data[each_col])

for each_col in str_cols_test_data:
    test_data[each_col] = le.fit_transform(test_data[each_col])

# Checking the training and testing data now.
print(train_data.info())
print(test_data.info())

# importing seaborn
import seaborn as sns

# creating heatmap
plt.figure(figsize=(14, 11))
sns.heatmap(train_data.corr(), annot=True)
plt.show()

# importing the library to perform VIF.
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Checking the list of columns once, in form of a list!
list(train_data.columns)

# Creating a list of feature columns.
feature_cols = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Title",
    "Title_group",
]

# Creating dataset with features.
X = train_data[feature_cols]

# Checking the shape of features data.
X.shape

# creating a new dataframe to store the variance inflation factor outcomes.
vif = pd.DataFrame()
vif["Features"] = feature_cols
vif["vif"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Checking the VIF dataset.

vif

# We will drop the Title column first since it has a very high value.
train_data.drop("Title", axis=1, inplace=True)
test_data.drop("Title", axis=1, inplace=True)

# Rechecking the VIF information
feature_cols = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Title_group",
]
X = train_data[feature_cols]
vif = pd.DataFrame()
vif["Features"] = feature_cols
vif["vif"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif

# Checking, finally, the data related to training and testing.
print(train_data.info())
print(test_data.info())

# Checking for outliers in the data.
for each_col in X.columns:
    plt.figure(figsize=(14, 12))
    plt.boxplot(train_data[each_col])
    plt.xlabel(each_col)
    plt.show()
