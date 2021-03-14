# This file is to solve the P1 - Credit Approval. Using Logistic regression model

from sklearn import linear_model
from logistic_utils import *
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression

# load the row data. I changed the crx.data into crx.csv, just add the column names
row_data = pd.read_csv("data/crx.csv")
x = delete_line("data/crx.csv")

# delete all the missing values
count = 0
for i in range(len(x)):
    row_data = row_data.drop(row_data.index[x[i] - count])
    count = count + 1

# insert the index
row_data.index = range(len(row_data))
# store usable data into 'new_data.csv'
row_data.to_csv("new_data.csv")

# load the usable data and set the label
new_df = pd.read_csv("new_data.csv")
new_df.loc[new_df['A16'] == '+', 'A16'] = 1
new_df.loc[new_df['A16'] == '-', 'A16'] = 0
# regularize data
df = regularized_data(new_df)

# filter the feature and label
all_data_df = df.filter(regex='A16|A1_.*|A2|A3|A4_.*|A5_.*|A6_.*|A7_.*|A8|A9_.*|A10_.*|A11|A12_.*|A13_.*|A14|A15')
# insert the y_hat into index 0
cols = list(all_data_df)
cols.insert(0,cols.pop(cols.index('A16')))
all_data_df = all_data_df.loc[:,cols]

# split all_data into train and test
train, test = train_test_split(all_data_df, test_size=0.4)

# training the model
train_np = train.values
y_train = train_np[:, 0]
y_train = y_train.astype('int')
X_train = train_np[:, 1:]
clf = LogisticRegression(penalty = 'none', max_iter=100, class_weight='balanced')
#clf = linear_model.LogisticRegression(solver='lbfgs', max_iter=100, penalty='none')
clf.fit(X_train,y_train)

# analysis parameters
pd.DataFrame({"columns":list(all_data_df.columns)[1:], "coef":list(clf.coef_.T)})

# test the model
test_np = test.values
y_test = test_np[:, 0]
y_test = y_test.astype('int')
X_test = test_np[:, 1:]
test_accurcy = clf.score(X_test, y_test)
print("Accuracy rate is: ", test_accurcy)

#predictions = clf.predict(y_test)

# simple cross-validation
all_data_np = all_data_df.values
# train_np
X = all_data_np[:,1:]
y = all_data_np[:,0]
y = y.astype('int')
print("The cross-validation accuracy rate is: \n", cross_val_score(clf, X, y, cv=5))

# Using learning curve to check the overfitting and underfitting of this model
# warning: when I use all the data to cross-validation
plot_learning_curve(clf, "learning_curve", X, y)