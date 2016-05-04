import csv as csv
import numpy as np
import gauss_naivebayes
from scipy.stats import itemfreq

def main():
    #Import Data
    csv_file_object = csv.reader(open('adult.data', 'rU'))
    ages = []
    salary = []
    edu = []
    for line in csv_file_object:
        ages.append(line[0])
        salary.append(line[14])
        edu.append(line[3])

    age = np.array(ages).astype('int')
    edu = np.array(edu).astype('str')
    salary = np.array(salary).astype('str')

    edu_trans = []
    for row in edu:
        key = row.strip()
        scores = {"Doctorate": 15,
        "Masters": 14,
        "Bachelors": 13,
        "Prof-school": 12,
        "Assoc-acdm": 11,
        "Assoc-voc": 10,
        "Some-college": 9,
        "HS-grad": 8,
        "12th": 7,
        "11th": 6,
        "10th": 5,
        "9th": 4,
        "7th-8th": 3,
        "5th-6th": 2,
        "1st-4th": 1,
        "Preschool": 0}

        edu_trans.append(scores[key])

    edu = np.array(edu_trans).astype('int')

    age = np.stack((age,edu), axis=-1)

    salary_trans = []
    for row in salary:
        row = row.strip()
        if row == "<=50K":
            x = 0
            salary_trans.append(x)
        elif row == '>50K':
            x = 1
            salary_trans.append(x)

    salary = np.array(salary_trans).astype('int')

    #Run Gaussian Naive Bayes
    gauss_result = gauss_naivebayes.gauss(age,salary)
    print gauss_result
    for i in range(18,65):
        print i, gauss_result.predict([[i, 14]])

    """
    position = np.where(age == 30)
    pay = []
    for i in position:
        x = salary[i]
        pay.append(x)
    print pay
    print itemfreq(pay)
    """

if __name__=="__main__":
    main()

