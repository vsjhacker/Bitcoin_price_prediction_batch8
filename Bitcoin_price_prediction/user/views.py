from django.shortcuts import render,redirect

# Create your views here.
from user.models import user_reg, bitcoin_price3
import numpy as np
import pandas as pd
from sklearn.svm import SVR
# Plotting graphs
import matplotlib

import matplotlib.pyplot as plt

# Machine learning libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def user_index(request):

    return render(request, 'user/user_index.html')

def user_login(request):
    if request.method == "POST":
        uname = request.POST.get('uname')
        pswd = request.POST.get('password')
        try:
            check = user_reg.objects.get(uname=uname, password=pswd)
            request.session['uid'] = check.id
            request.session['uname'] = check.uname

            return redirect('user_home')
        except:
            pass
        return redirect('user_login')
    return render(request, 'user/user_login.html')

def user_register(request):
    if request.method == "POST":
        fullname = request.POST.get('fullname')
        email = request.POST.get('email')
        mobile = request.POST.get('mobile')
        uname = request.POST.get('uname')
        password = request.POST.get('password')

        user_reg.objects.create(fullname=fullname, email=email, mobile=mobile, uname=uname, password=password)
        return redirect('user_login')
    return render(request, 'user/user_register.html')


def user_home(request):
    bitcoin_price=bitcoin_price3.objects.all()
    return render(request, 'user/user_home.html',{'bitcoin_price':bitcoin_price})


def knn(request):
    df = pd.read_csv('bitcoin_price.csv')

    df = df.dropna()
    df = df[['Open', 'High', 'Low', 'Close']]   #taking out these cols.
    print(df.head())

    df['Open-Close'] = df.Open - df.Close     #create col
    df['High-Low'] = df.High - df.Low
    df = df.dropna()
    X = df[['Open-Close', 'High-Low']]
    print(X.head(50))

    Y = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)  # ..


    # Splitting the dataset
    split_percentage = 0.7
    split = int(split_percentage * len(df))

    X_train = X[:split]  # open-close high-low
    Y_train = Y[:split]   # 1 -1 1 -1

    X_test = X[split:]
    Y_test = Y[split:]

    # Instantiate KNN learning model(k=15)
    knn = KNeighborsClassifier(n_neighbors=15)  #15 members

    # fit the model
    knn.fit(X_train, Y_train)       # data set and class result

    # Accuracy Score
    accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
    accuracy_test = accuracy_score(Y_test, knn.predict(X_test))

    print('Train_data Accuracy: %.2f' % accuracy_train)
    print('Test_data Accuracy: %.2f' % accuracy_test)

    df['Predicted_Signal'] = knn.predict(X)
    print(df[['Predicted_Signal']].head(1250))

    #  Cumulative Returns

    df['SPY_returns'] = np.log(df['Close'] / df['Close'].shift(1))         # Natural_Log  logE(today/yesterday)
    Cumulative_SPY_returns = df[split:]['SPY_returns'].cumsum() * 100         # applying for last 30% selling purchase
    print(Cumulative_SPY_returns.head(100))


    # Cumulative Strategy Returns
    df['Strategy_returns'] = df['SPY_returns'] * df['Predicted_Signal'].shift(1)
    Cumulative_Strategy_returns = df[split:]['Strategy_returns'].cumsum() * 100
    print(Cumulative_Strategy_returns.head(100))


    # Plot the results to visualize the performance

    plt.figure(figsize=(10, 5))
    plt.title('Cumulative Returns')
    plt.plot(Cumulative_SPY_returns, color='r', label='cumulative Returns')
    plt.plot(Cumulative_Strategy_returns, color='g', label='Strategy Returns')
    plt.legend()
    plt.show()
    return render(request, 'user/knn.html',{'accuracy_train':accuracy_train,'accuracy_test':accuracy_test})



def svm(request):
    df = pd.read_csv('bitcoin_price1.csv')
    print(df.head(7))

    # Remove the Date column
    df.drop(['Date'], 1, inplace=True)

    # Show the first 7 rows of the new data set
    print(df.head(7))

    # A variable for predicting 'n' days out into the future
    prediction_days = 30  # n = 30 days

    # Create another column (the target or dependent variable) shifted 'n' units up
    df['Prediction'] = df[['Close']].shift(-prediction_days)  # minus 30

    print(df.head(7))

    # Show the last 7 rows of the new data set
    print(df.tail(7))

    # CREATE THE INDEPENDENT DATA SET (X)

    # Convert the dataframe to a numpy array and drop the prediction column
    X = np.array(df.drop(['Prediction'], 1))   # converting into array without first row and dropping

    # Remove the last 'n' rows where 'n' is the prediction_days
    X = X[:len(df) - prediction_days]
    print(X)

    # CREATE THE DEPENDENT DATA SET (y)
    # Convert the dataframe to a numpy array (All of the values including the NaN's)
    y = np.array(df['Prediction'])
    # Get all of the y values except the last 'n' rows
    y = y[:-prediction_days]  #except last 30 -operator.
    print(y)

    # Split the data into 80% training and 20% testing
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Set prediction_days_array equal to the last 30 rows of the original data set from the price column
    prediction_days_array = np.array(df.drop(['Prediction'], 1))[-prediction_days:]
    print(prediction_days_array)

    # Create and train the Support Vector Machine
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001) #radial basis function
    # Create the model
    svr_rbf.fit(x_train, y_train) #preprocessing
    # Train the model

    # Testing Model: Score returns the accuracy of the prediction.
    # The best possible score is 1.0
    svr_rbf_confidence1 = svr_rbf.score(x_train, y_train)
    svr_rbf_confidence = svr_rbf.score(x_test, y_test)
    print("svr_rbf accuracy: ", svr_rbf_confidence)
    print("Train data Accuracy: ", svr_rbf_confidence1)

    # Print the predicted value
    svm_prediction = svr_rbf.predict(x_test)
    print(svm_prediction)

#    print()


    # Print the actual values
 #   print(y_test)*/

    # Print the model predictions for the next 'n=30' days
    svm_prediction = svr_rbf.predict(prediction_days_array)
    print(svm_prediction)

    print(df.tail(prediction_days))

    plt.plot(svr_rbf.predict(x_test), c='r', label='RBF model')

    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    return render(request, 'user/svm.html',{'accuracy_train':svr_rbf_confidence1,'accuracy_test':svr_rbf_confidence})
