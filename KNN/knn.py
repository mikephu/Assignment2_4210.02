#-------------------------------------------------------------------------
# AUTHOR: Michael Phu
# FILENAME: knn.py
# SPECIFICATION: Reads training/test data from .csv files and utilizes the sklearn library to classify instances and verify its predictions using the 1NN method. It displays the 
# LOO-CV error rate of the algorithm's performance on the testing data.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 30 Mins (KNN)
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

# Map classes to numbers
classes = {'-':1,'+':2}

# Keeping track of wrong predictions
wrongPredict = 0 

#loop your data to allow each instance to be your test set
for i, instance in enumerate(db):

    X = [] 
    Y = []
    #add the training features to the 2D array X and remove the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 3], [2, 1,], ...]]. Convert values to float to avoid warning message
    for row in db: 
        if row != instance:
            X.append([float(row[0]),float(row[1])])

    #transform the original training classes to numbers and add them to the vector Y. Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...]. Convert values to float to avoid warning messages
   
    for row in db: 
        if row != instance:
            Y.append(float(classes[row[2]]))

    #--> add your Python code here
    testSample = instance

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    class_predicted = clf.predict([[float(testSample[0]),float(testSample[1])]])[0]

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    if class_predicted != classes[testSample[2]]:
        wrongPredict += 1

#print the error rate
print("LOO-CV Error Rate for 1NN:",(wrongPredict/len(db)))






