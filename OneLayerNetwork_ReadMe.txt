The sigmoid function is-

S = (1/(1 + np.exp(-x)))



And its derivative is-

dS/dx = ( np.exp(-x)  /  (1 + np.exp(-x))**2   )

	which can be rewritten as -

	dS/dx = (1/(1 + np.exp(-x)))  *   ( 1  -  (1/(1 + np.exp(-x))) )

	or

dS/dx = S * (1 - S)
