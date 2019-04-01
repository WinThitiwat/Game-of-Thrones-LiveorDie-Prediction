# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:19:55 2019

@author: Thitiwat W. (Win)
"""
# Loading Libraries
import pandas as pd


# loading data
GOT = pd.read_excel("GOT_character_predictions.xlsx")

#############################################
# explore the data
#############################################
print(GOT.shape)

print(GOT.head(15))

print(GOT.info())

print(GOT.isnull().sum())

GOT = GOT.drop([
    "S.No",
    "father",
    "mother",
    "heir",
    "isAliveMother",
    "isAliveFather",
    "isAliveHeir",
    "isAliveSpouse",
    "spouse",
    "title"], axis=1)

#############################################
# check remaining missing values columns and impute those missing values
#############################################

# check `age` distribution as it is integer datatype
GOT.boxplot(['age'])

# it seems like there are 2 outliers with weird data, which -298007 & -277980
# after research character from those 2 observations, it seems like the
# first 3 digits are the age of those character
GOT.loc[GOT['age'] == -298001.0, 'age'] = 298
GOT.loc[GOT['age'] == -277980.0, 'age'] = 278

# check `dateOfBirth` distribution as it is integer datatype
GOT.boxplot(['dateOfBirth'])

# it seems like there are 2 outliers with weird data, which 278279 & 298299
# after research character from those 2 observations, it seems like the
# first 3 digits are the year of birth of those characters after Jesus Christ
GOT[GOT['dateOfBirth'] == 298299]['dateOfBirth'] = 298
GOT[GOT['dateOfBirth'] == 278279]['dateOfBirth'] = 278

# impute missing value in categorical feature with 'Unknown'
# and age with median after remove outliers
m_cols = {
    'title': 'Unknown',
    'house': 'Unknown',
    'age': GOT['age'].median(),  # median after remove outliers
    'culture': 'Unknown',
    'dateOfBirth': GOT['dateOfBirth'].median()
}
GOT.fillna(value=m_cols, inplace=True)

#############################################
# feature engineerign
#############################################

# initial new column with 0
GOT['great_house'] = 1

# create a list of great house based on the story from Game of Thrones
great_house = [
        "Arryn",
        "Greyjoy",
        "Lannister",
        "Stark",
        "Targaryen",
        "Tully",
        "Frey",
        "Casterly",
        "Mudd",
        "Justman",
        "Hoare",
        "Durrandon",
        "Gardener",
        "Baratheon",
        "Martell",
        "Bolton",
        "Tyrell" 
        ]

# make a copy and query a series of `house` columns
GOT_house = GOT['house'].copy()
GOT.loc[GOT['house']=="Unknown", 'great_house'] = -1
# loop the `great_house` to flag houses from `GOT_house` if in those list
for each_house in great_house:
    gh_idx = GOT_house.str.lower().str.contains(
        each_house.lower(),
        regex=False
        )
    if gh_idx.any():
        GOT.loc[gh_idx, "great_house"] = 0

# check total number of great house in the dataset
GOT["great_house"].value_counts(dropna=False)



# create new feature representing rows
# in which `house` is 'Unknown' or initially missing
GOT["m_house"] = 1
GOT_unknown_house_df = GOT[GOT['house'] == "Unknown"]
GOT.loc[GOT_unknown_house_df.index, "m_house"] = 0
GOT.shape
# export clean data
GOT.to_excel("cleaned_GOT.xlsx", index=False)
