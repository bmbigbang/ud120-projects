#!/usr/bin/python

import operator

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = sorted([(j, k, (i-k)**2) for i, j, k in zip(predictions, ages, net_worths)], key=operator.itemgetter(2))

    ### your code goes here

    return cleaned_data[:-len(cleaned_data)/10]

