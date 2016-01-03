import csv as csv
import numpy as np
import gauss_naivebayes
from scipy.stats import itemfreq


csv_file_object = csv.reader(open('adult.data', 'rU'))

ages = []
outcome = []
edu = []
for line in csv_file_object:
	ages.append(line[0]) #put back sq brackets about line[0]
	outcome.append(line[14])
	edu.append(line[3])

datax = np.array(ages).astype('int')
datax2 = np.array(edu).astype('str')
datay = np.array(outcome).astype('str')

print datax
print datax2

datax2_trans = []
for row in datax2:
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

	datax2_trans.append(scores[key])

datax2 = np.array(datax2_trans).astype('int')

datax = np.stack((datax,datax2), axis=-1)

datay_trans = []
for row in datay:
	row = row.strip()
	if row == "<=50K":
		x = 0
		datay_trans.append(x)
	elif row == '>50K':
		x = 1
		datay_trans.append(x)

datay = np.array(datay_trans).astype('int')

gauss_result = gauss_naivebayes.gauss(datax,datay)
print gauss_result

"""
position = np.where(datax == 30)
pay = []
for i in position:
	x = datay[i]
	pay.append(x)
print pay
print itemfreq(pay)

"""
for i in range(18,65):
	print i, gauss_result.predict([[i, 14]])
