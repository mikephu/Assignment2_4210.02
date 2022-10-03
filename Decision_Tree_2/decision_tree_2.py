#-------------------------------------------------------------------------
# AUTHOR: Michael Phu 
# FILENAME: decision_tree_2.py
# SPECIFICATION: This program utilizes the sklearn library to create a model (decision tree) for each of the data sets contained in the training .csv files 10 times. 
# The program will use the testing data / .csv file to calculate the accuracy of each model after it has been built, in which it displays the lowest accuracy over the 10 iterations for each file.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 Hour (decision_tree_2)
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    features = {'Young':1,'Prepresbyopic':2,'Presbyopic':3,'Myope':1,'Hypermetrope':2,'Yes':1,'No':2,'Reduced':1,'Normal':2}
    for i in range(0,len(dbTraining)):
        curRow = [] 
        for j in range(0,len(dbTraining[0])-1):
            if dbTraining[i][j] in features: 
                curRow.append(features[dbTraining[i][j]])
        X.append(curRow)

    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    classes = {'Yes':1,'No':2}
    for i in range(0,len(dbTraining)):
        Y.append(classes[dbTraining[i][4]])

    accuracy = 1
    #loop your training and test tasks 10 times here
    for i in range (10):
       #fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
       clf = clf.fit(X, Y)

       #read the test data and add this data to dbTest
       #--> add your Python code here
       dbTest = [] 
       with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append(row)
       
       fn,fp,tp,tn = 0,0,0,0

       for data in dbTest:
           #transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           features = {'Young':1,'Prepresbyopic':2,'Presbyopic':3,'Myope':1,'Hypermetrope':2,'Yes':1,'No':2,'Reduced':1,'Normal':2}
           testX = []
           classY = features[data[4]]
           for j in range(0,len(data)-1):
                testX.append(features[data[j]])
           class_predicted = clf.predict([testX])[0]
           
           #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           if class_predicted != classY: 
                if classY==1:
                    fn+=1
                else:
                    fp+=1
           else:
                if classY==1:
                    tp+=1
                else:
                    tn+=1
        #find the lowest accuracy of this model during the 10 runs (training and test set)
       accuracy = min(accuracy,((tp+tn)/(tp+tn+fn+fp)))
    
    # print the lowest accuracy of this model during the 10 runs (training and test set).
    # your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print('Final accuracy when training on',ds + ':',accuracy)




