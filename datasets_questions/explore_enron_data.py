#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# Lay Kenneth payment
for i in enron_data:
    if "lay" in i.lower() and "kenneth" in i.lower():
        for j in enron_data[i]:
            print j, enron_data[i][j]
    else:
        continue

# counter for emails/salary
# counter = 0
# for i in enron_data:
#     if enron_data[i]['email_address'] and enron_data[i]['email_address'] != "NaN":
#         print enron_data[i]['email_address']
#         counter += 1
# print counter

# counter for PoI with NaN payment totals
# counter = 0
# for i in enron_data:
#     if enron_data[i]['poi'] and enron_data[i]['total_payments'] == "NaN":
#         counter += 1
#
# print counter + 10, (counter + 10) / (float(len(enron_data)) + 10)

