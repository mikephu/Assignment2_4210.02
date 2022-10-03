#-------------------------------------------------------------------------
# AUTHOR: Michael Phu 
# FILENAME: naive_bayes.py
# SPECIFICATION: Reads training and test data from csv files and using sklearn's naive bayes tools to display class instance predictions with a confidence of >= 0.75.
# FOR: CS 4210 - Assignment #2
# TIME SPENT: 45 Mins (naive_bayes)
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv 

db = []
dbTest = []
X = [] 
Y = []

#reading the training data in a csv file
with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append(row)

features = {'Sunny':1,'Overcast':2,'Rain':3,'Hot':1,'Mild':2,'Cool':3,'Normal':1,'High':2,'Weak':1,'Strong':2}
classes = {'Yes':1,'No':2}

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
for row in db: 
  curRow = [] 
  for i in range(0,len(row)):
    if row[i] in features: 
      curRow.append(features[row[i]])
    elif row[i] in classes: 
      Y.append(classes[row[i]])
  X.append(curRow)

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
with open('weather_test.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         dbTest.append(row)

#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
testX = []
for row in dbTest: 
  curRow = [] 
  for i in range(0,len(row)):
    if row[i] in features: 
      curRow.append(features[row[i]])
  testX.append(curRow)

for i in range(0,len(dbTest)):
    confidence = list(clf.predict_proba([testX[i]])[0])
    if confidence[0] > 0.75 or confidence[1] > 0.75:

      if confidence[0] > 0.75: 
        probability = confidence[0]
        classInstance = 'Yes'
      else: 
        probability = confidence[1]
        classInstance = 'No'
    
      print(dbTest[i][0].ljust(15) + dbTest[i][1].ljust(15) + dbTest[i][2].ljust(15) + dbTest[i][3].ljust(15) + dbTest[i][4].ljust(15) + 
      classInstance.ljust(15) + ((str(round(float(probability),2))).ljust(15)))
    


