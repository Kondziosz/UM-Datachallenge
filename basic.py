#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

all_data = pd.concat([train_data, test_data], ignore_index=True)
#%%
def print_missing_and_zero_values(df: pd.DataFrame):
    # Calculate the number of missing values for each column
    missing_values = df.isnull().sum()

    # Print the number of missing values for each column
    print("Number of null values for each column:")
    print(missing_values)

    # Calculate the number of zero values for each column
    zero_values = (df == 0).sum()

    # Print the number of zero values for each column
    print("\nNumber of '0' values for each column:")
    print(zero_values)

print_missing_and_zero_values(all_data)
# %%
age_median = all_data['Age'].median()
all_data['Age'].fillna(age_median, inplace=True)
all_data['Embarked'].fillna(all_data['Embarked'].mode()[0], inplace=True)
all_data["Fare"].fillna(0, inplace=True)

#median_fares = all_data.groupby('Pclass')['Fare'].median()
# Calculate the count of each unique ticket
all_data['TicketCount'] = all_data.groupby('Ticket')['Ticket'].transform('count')

# Filter the data for entries that have a 'TicketCount' of 1
single_ticket_data = all_data[all_data['TicketCount'] == 1]

# Calculate the median fare per ticket class for these entries
median_fares_single_ticket = single_ticket_data.groupby('Pclass')['Fare'].median()
all_data['Fare'] = all_data.apply(lambda row: median_fares_single_ticket[row['Pclass']] if row['Fare'] == 0 else row['Fare'], axis=1)

all_data['Sex'] = all_data['Sex'].map({'male': 0, 'female': 1})

# Create a new column 'PricePerPerson' which is the 'Fare' divided by 'TicketCount'
all_data['PricePerPerson'] = all_data['Fare'] / all_data['TicketCount']
# %%
all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# Replace 'Mme' and 'Mlle' titles with 'Miss'
all_data['Title'] = all_data['Title'].replace(['Mme', 'Mlle'], 'Miss')
all_data['Title'] = all_data['Title'].replace(['Lady', 'Ms', 'The Countess', 'Countess', 'Dona'], 'Mrs')
all_data['Title'] = all_data['Title'].replace(['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir'], 'Mr')

all_data.drop(['Name', 'Title', 'Cabin', 'Ticket', 'Embarked'], axis=1, inplace=True)
# %%
train_data = all_data[all_data['Survived'].notnull()]
test_data = all_data[all_data['Survived'].isnull()]

sns.heatmap(train_data.corr(), cmap="YlGnBu")
plt.show()