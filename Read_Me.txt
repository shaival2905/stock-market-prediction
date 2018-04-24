There are 3 models which i have made. 2 for classification and 1 for regression

I am using keras library which is build above tensorflow for creating machine learning models.

Features:
I have calculated 11 statistical features for my input.
I am training my model in such a way that the trend of the next day is taken as target value for current day. So my model will predict the trend for next day.
Similarly for regression close price of next day is selected as a target for current day.

1) Artificial neural network is taken as classifier in one model
It has 3 layers. 
2 layers uses relu activation function and last layer uses sigmoid.
Main reason behind this is relu function creates more sparse data when W*X + B < 0 and other reason is it reduces chances of the gradient to vanish. Using relu function faster learning is done. 

taking sigmoid or relu activation function in output layer i.e last layer gives almost same accuracy so I have taken sigmoid function to obtain non linear model. Also I have normalized the dataset.

The accuracy got in ANN model is 69.77%

2) Other classifier used is Support Vector Machine(SVM)
It is created using scikit learn library.
SVM is not better than neural network.
SVM computes faster as it is like shallow neural network which has only one layer.
The main reason to compute this model is to use this to create a hybrid model with ANN for better results. 

The accuracy got in SVM is 51%

3) Support vector regression is created for predicting stock prices.
It is also created using scikit learn library.
Linear fitting and polynomial fitting is compared. Linear fitting is giving less mean error.

Mean error of polynomial:  1.02668038
Mean error of linear:  0.6860455

graph is plot on predicted price and actual prize to compare.

I am focusing more on calculating the and finding features which affects the stock prices. Right now i have only calculated 11 features using moving average for 10 days. 
Features are mentioned in the code
I have selected these features on the basis of literature survey of research papers.  