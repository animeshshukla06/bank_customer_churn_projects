from flask import Flask, render_template, request
import numpy as np
import pickle
import mysql.connector

# store the user's data in the database:
# def store_data(form_data_dict, pred_class):
#   list_values_form = list(form_data_dict.values())
#    list_values_form.append(pred_class)
#    tuple_values_form = tuple(list_values_form) # since execute() method takes tuple as arguments

#     defining connection parameters / configerations of MariaDB database:
#    db_config = {
     #   'host': '127.0.0.1',
      #  'user': 'bank_churn_username',
       # 'password': 'bank_churn_password',
        #'database': 'bank_churn_db'
   # }

    # establishing a connection between Python and the database:
    #conn = mysql.connector.connect(**db_config)  # keyword arguments(config of database) given as dictionary
    #cursor = conn.cursor() # used for sending and retrieving the data:
    #_SQL = """ INSERT INTO customers_data 
     #           (CreditScore, Gender, Age, Balance, NumOfProducts, IsActiveMember, Country, Exited)
      #          VALUES
       #         (%s, %s, %s, %s, %s, %s, %s, %s)"""
    #cursor.execute(_SQL, tuple_values_form)
    #conn.commit() # making sure that the data gets stored in the database.
    #cursor.close() # since we have limited resources, so we always close the cursor.
    #conn.close() # henceforth, we close the connection as well.
            

# function to give the predictions:
# arguments new_dict is fully encoded and now we simply need to train the model:
def predictions_proba(dict_prepr_data):
    data = np.array(list(dict_prepr_data.values())).reshape(1, 9) #converting the values to array
    model = pickle.load(open('model_final.pkl', 'rb')) # loading the saved svm model weights
    predictions = model.predict_proba(data) # predicting probability of each class
    pred_class = model.predict(data) # predicting target class 
    if pred_class[0] == 1:
        return [predictions[0][0], predictions[0][1], 'Leave']
    else:
        return [predictions[0][0], predictions[0][1], 'Continue']


# function used to preprocess the form data based upon the min-max-scaler:
def preprocessing_data(dict_data):
    # values derived from the training set:
    # each feature has its first value as max value and the next value of list is min value.
    # feature --> [max_val, min_val]
    train_max_min_val = {'creditscore': [850.0, 381.0],
                        'gender': [1, 0], 'age': [64.0, 18.0], 
                        'balance': [250898.09, 0.0], 
                        'num_of_products': [4, 1], 'activemember': [1, 0], 
                        'France': [1, 0], 'Germany': [1, 0], 'Spain': [1, 0]}
    
    pre_pr_dict = {} # empty dictionary for containing pre-preocessed data
    # min max scaling on the data recieved from the website:
    for key in dict_data:
        range_val = train_max_min_val[key][0] - train_max_min_val[key][1]
        pre_pr_dict[key] = ( int(dict_data[key]) - train_max_min_val[key][1] ) / range_val

    return pre_pr_dict

lst_data_form = []
# # to shape the inputs as required by the model:
def shaping_inputs(my_dict):
    global lst_data_form
    for key in my_dict:

        if key not in (['gender', 'activemember', 'country']): #because three of them need decoding:
            lst_data_form.append((key, my_dict[key]))

        else:

            # encoding for gender:
            # Mapping the Categories of Gender as: Female--> 0, Male-->1
            if key == 'gender':
                if my_dict['gender'] == 'Female':
                    lst_data_form.append(('gender', 0))
                else:
                    lst_data_form.append(('gender', 1))
            
            # encoding for activemember
            if key == 'activemember':
                if my_dict['activemember'] == 'Yes':
                    lst_data_form.append(('activemember', 1))
                else:
                    lst_data_form.append(('activemember', 0))

            # encoding for country:
            if key == 'country':
                if my_dict['country'] == 'France' or my_dict['country'] == 'france':
                    lst_data_form.extend([('France', 1), ('Germany', 0), ('Spain', 0)])
                if my_dict['country'] == 'Germany' or my_dict['country'] == 'germany':
                    lst_data_form.extend([('France', 0), ('Germany', 1), ('Spain', 0)])
                if my_dict['country'] == 'Spain' or my_dict['country'] == 'spain': 
                    lst_data_form.extend([('France', 0), ('Germany', 0), ('Spain', 1)])    
    
    dict_prepr_data = preprocessing_data(dict(lst_data_form)) # callng preprocessing_data function:
    [prob_0, prob_1, pred_class] = predictions_proba(dict_prepr_data) # calling the function which computes probability.
    return [prob_0, prob_1, pred_class]


# instace of Flask which requires the current module as __name__
app = Flask(__name__) 

# 2 URL's have been attached to the same function: (could have also used redirect)
@app.route('/')
@app.route('/home')
def hello():
    return render_template('home.html') # just displays the web-page.

# method = ['POST'] means the form data will be sent over the web-server in a secured manner.
# data is provided from the managers of the bank which is first encoded (for categorical data)
# and then preprocessed using min-max-scaler. Further it computes the probability of that data
# belonging to each class and also predicts the target_class(0-> Continuing or 1->leaving). 
@app.route('/results', methods=['POST'])
def results():
    form_data_dict = request.form # form data as dictionary.
    [prob_0, prob_1, pred_class] = shaping_inputs(form_data_dict) # preprocessing the data.
    # store_data(form_data_dict, pred_class) # sending the data of the form to get stored in the database.
    return render_template('results.html', 
                            prob_0 = str(round(prob_0*100, 3)) + '%',
                            prob_1 = str(round(prob_1*100, 3)) + '%',
                            target_class = pred_class)

# it is used to enusure whether the python file is being run directly or not.
# if this file is imported into another python file, then the app would not run, 
# since then __name__ == 'name of the file'.
if __name__ == '__main__':
    app.run(debug=True)


# example data from the form:
# Hence the verification of the code is done, everything is working fine:
# my_dict = {'creditscore': 670, 'gender': 'Male', 'age': 65,
#  'balance': 0, 'num_of_products': 2, 'activemember': 'Yes', 
#  'country':'France'}

#print(shaping_inputs(my_dict))

# [prob_0, prob_1, pred_class] = shaping_inputs(my_dict) # preprocessing data:
# print('Probability of the customer not leaving the bank', prob_0)
# print('Probability of the customer leaving the bank', prob_1)
# print('Predicted Class--->', pred_class)

# C:\Users\DELL\Python\Scripts\Jupyter Notebooks\Projects Notebooks\Churning of Customer\webapp
