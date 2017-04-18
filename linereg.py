#Importing the required libraries
import sys
import numpy as np
from pyspark import SparkContext


#Function Definition

# To get the Matrix Y
def Value_of_Y(value):
    return [value[0]]

 
# To get the Matrix X
def Value_of_X(value):
    value[0]=1.0
    return [value]

#Spark program execution start
if __name__ == "__main__":
    
     #If input file is not provided, show error
    if len(sys.argv) !=4:
        print >> sys.stderr, "Usage: linreg <datafile> <step size> <iterations>"
        exit(-1)
    #Initiatlize spark context
    sc = SparkContext(appName="LinearRegression")
    #Read the entire input file on to rdd and perform map operation by splitting on comma(,)
    yxinputFile = sc.textFile(sys.argv[1])
    yxlines = yxinputFile.map(lambda line: line.split(','))
    #yxlength = len(yxfirstline)
		
    # Get the values of Matrix Y and X
    Y = np.asmatrix(yxlines.map(lambda n: ('Value_of_Y',Value_of_Y(n))).reduceByKey(lambda v1,v2: v1+v2).map(lambda n:n[1]).collect()[0]).astype(float).T
    X = np.asmatrix(yxlines.map(lambda n: ('Value_of_X',Value_of_X(n))).reduceByKey(lambda v1,v2: v1+v2).map(lambda n:n[1]).collect()[0]).astype(float)

    Number_of_betas = X.shape[1]
    #print("Number_of_betas",Number_of_betas)

    beta_initial = []
    num=0
    while(num < Number_of_betas):
        beta_initial.append(1)
        num=num+1
    ######Gradient Decent######
    num_of_iterations= int(sys.argv[3])
    #num_of_iterations = 3000
    #Step size alpha  
    alpha = float(sys.argv[2])
    #alpha = 0.0001
    #import pdb; pdb.set_trace()

    # converting to matrix
    Matrix_beta_initial = []
    for val in beta_initial:
        Matrix_beta_initial.append(val)
    n = 0   
    #for iter in range(0,iterations):
    while(n < num_of_iterations):
        if n == 0:
            beta_last_iteration = Matrix_beta_initial
              
        beta = np.matrix(beta_last_iteration).T
        xbeta = np.dot(X,np.matrix(beta))
        ysubxbeta = np.subtract(Y,xbeta)
        newBeta = (np.add(beta,np.multiply(np.dot(X.T,ysubxbeta),alpha)))
        beta_curr_iteration = []
        for val in newBeta.tolist():
            beta_curr_iteration.append(val[0])
        print ("beta after iternations: " + str(n+1))
        for coefficient in beta_curr_iteration:      
            print (coefficient)
		# Break the loop once beta converges
        if (beta_last_iteration == beta_curr_iteration ):
            print("beta converged in "+ str(n+1)+ " iternations")
            break
        beta_last_iteration = beta_curr_iteration
        n = n + 1

        
