import numpy as np

#Here we have a 1 to 1 linear relationship between input's 1st column and output
#So it works with a single neural layer
input_data = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
output_data = np.array([[0,1,1,0]]).T


#Objective function(also contains Loss function, which is called when arrIsError is true)
#It takes any input (number or array) and maps it to a (number or array with elements) between 0 & 1
def objectiveFxn(arr, arrIsError = False):
	if arrIsError == True:
		#Below expression is derivative of sigmoid function,
		#but arr here, will already have values between 0 & 1, 
		#so we replaced sigmoid function with arr
		return (arr * (1 - arr))
	#Sigmoid function
	return (1/(1 + np.exp(-arr)))


weights = np.random.random((3,1))

for i in range(1000000):
	predicted_data = objectiveFxn(np.dot(input_data, weights))

	error_data = output_data - predicted_data

	if i % 200000 == 0:
		print ('error after '+str(i)+ ' iterations : ')
		print (error_data)

	gradient = np.dot(input_data.T , error_data * objectiveFxn(predicted_data, arrIsError = True))

	weights += gradient

#Testing
print('Final weight:')
print (weights)
print('prediction for 0,0,0 :')
print (objectiveFxn(np.dot(np.array([0,0,0]), weights)))
print('prediction for 1,1,1 :')
print (objectiveFxn(np.dot(np.array([1,1,1]), weights)))