#Importing the required libraries
import sys
import numpy as np
from pyspark import SparkContext


#Function Definition

# To multiply x and x-transpose
def Product_of_x_xt(value):
    value[0]=1.0
    X = np.asmatrix(value).T.astype('float')
    XT = X.T
    return np.dot(X,X.T)

# To get the product of Xt and y
def product_of_xt_y(value):
    y = float(value[0])
    value[0] = 1.0
    Xt = np.asmatrix(value).T.astype('float')
    return np.multiply(Xt,y)

 

#Spark program execution start
if __name__ == "__main__":
    
     #If input file is not provided, show error
    if len(sys.argv) !=2:
        print >> sys.stderr, "Usage: linreg <datafile> "
        exit(-1)
    #Initiatlize spark context
    sc = SparkContext(appName="LinearRegression")
    #Read the entire input file on to rdd and perform map operation by splitting on comma(,)
    yxinputFile = sc.textFile(sys.argv[1])
    yxlines = yxinputFile.map(lambda line: line.split(','))
    #yxlength = len(yxfirstline)
		
    # For computing the value of beta, calulating X_XT and Xt_y
    A = yxlines.map(lambda value: ("Product_of_x_xt",Product_of_x_xt(value))).reduceByKey(lambda xa,xb: np.add(xa,xb)).map(lambda x: x[1])
 
    #For each row, calculate the product of x-transpose and y
    
    B =  yxlines.map(lambda value: ("product_of_xt_y",product_of_xt_y(value))).reduceByKey(lambda xa,xb: np.add(xa,xb)).map(lambda x: x[1])
   

    #Calculating beta by A inverse B
    product = np.dot(np.linalg.inv(A.collect()),B.collect())[0]
    #covert the product into list
    beta_initial = np.array(product).tolist()
        
    #Print the beta values as per the required format
    print ("beta: ")
    for coefficient in beta_initial:      
        print (coefficient)
 
    

        
