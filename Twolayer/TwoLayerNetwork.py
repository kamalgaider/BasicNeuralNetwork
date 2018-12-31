import numpy as np

#Here we have a non linear relationship between the inputs and output
#So we first have to create a representaion of input data(combined value of all 3 columns in a row)
#& then create a 1 to 1 relationship with output
input_data = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
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


#we need this input to output transformation : 4*3 -> 4*1. Hence we selected below weight shapes-
weights_1 = np.random.random((3,4)) # 4*3  .  (3*x)  -> 4*x
weights_2 = np.random.random((4,1)) # 4*x  .  (x*1)  -> 4*1

for i in range(10000):
	Layer1_prediction = objectiveFxn(np.dot(input_data, weights_1))
	Layer2_prediction = objectiveFxn(np.dot(Layer1_prediction, weights_2))

	error_2 = output_data - Layer2_prediction
	gradient_2 = error_2 * objectiveFxn(Layer2_prediction, arrIsError = True)
	weights_2 += np.dot(Layer1_prediction.T, gradient_2)

	error_1 = np.dot(gradient_2, weights_2.T)
	gradient_1 = error_1 * objectiveFxn(Layer1_prediction, arrIsError = True)
	weights_1 += np.dot(input_data.T , gradient_1)

#Testing
print('Final weights:')
print (weights_1)
print (weights_2)
print('prediction for 0,0,0 :')
print (objectiveFxn(np.dot(objectiveFxn(np.dot(np.array([0,0,0]), weights_1)), weights_2)))
print('prediction for 1,1,0 :')
print (objectiveFxn(np.dot(objectiveFxn(np.dot(np.array([1,1,0]), weights_1)), weights_2)))