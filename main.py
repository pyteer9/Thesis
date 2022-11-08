import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import metodi_utili
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


# Reading the dataset
train_data = pd.read_csv('data/UNSW_NB15_training-set.csv')
test_data = pd.read_csv('data/UNSW_NB15_testing-set.csv')

train_len = len(train_data)
test_len = len(test_data)

'''
###########################################################

DATA PREPROCESSING

###########################################################
'''
# Removing unuseful features
test_data.drop(['id'], axis=1, inplace=True)
test_data.drop(['label'], axis=1, inplace=True)

train_data.drop(['id'], axis=1, inplace=True)
train_data.drop(['label'], axis=1, inplace=True)



# Merging together train and test csv file
dataframe = [train_data, test_data]
data = pd.concat(dataframe, ignore_index=True)


# Dropping off null values
data.dropna(inplace=True)

# Check that there aren't missing values after the cleaning step
print("The number of missing values is: ",
      data.isna().sum().sum())

'''
###################################################
One Hot encoding sulle feature: state, service e proto
###################################################
'''
## Effettuo il one hot encoding
data = pd.get_dummies(data=data, columns=['proto', 'service', 'state'])

'''
#######################################################################
NORMALIZZAZIONE Z-SCORE
#######################################################################
'''

# Drop delle variabili categoriche
attack_cat = data.pop('attack_cat')
print(data.describe())

# Z-score normalization
st = StandardScaler()
data = pd.DataFrame(st.fit_transform(data), columns=data.columns)

#Adding the class column to the dataframe
data["Class"] = attack_cat
'''
#######################################################################
MODEL IMPLEMENTATION
#######################################################################
'''
y_train = data["Class"]
data_X = data.drop(["Class"],axis=1)


ros = RandomOverSampler(sampling_strategy='minority')

k_value = 2
kfold = StratifiedKFold(n_splits=k_value, shuffle=True, random_state=42) # MOD: was 6 split
kfold.get_n_splits(data_X,y_train)
print(kfold)

model = metodi_utili.LSTM_model(data_X)

#Inizializing RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)

loop = True
while loop:
    choose = input("Insert: \n\t1) LSTM Model\n\t2) Random Forest Classifier\n\n\t--> ")
    if choose == "1":
        oos_pred, y_eval, pred = metodi_utili.LSTM_application(model, kfold, ros, data, data_X, y_train)
        metodi_utili.show_confusion_matrix(y_eval,pred, "Confusion Matrix LSTM Model")

    elif choose == "2":
        oos_pred, y_eval, pred = metodi_utili.RandomForest_Model(clf,kfold, ros, data, data_X, y_train)
        metodi_utili.show_confusion_matrix(y_eval, pred,"Confusion Matrix Random Forest Classifier")

    scelta = input("Do you wanna do another classification? \ny or n?\n--> ")
    if scelta != "y":
        loop = False



