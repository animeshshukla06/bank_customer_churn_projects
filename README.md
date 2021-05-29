<h1>Bank Customer Churn Project:</h1>
The aim of the project is to classify the customers of the bank based upon their personal details like 'Credit Score', 'Bank Balance', etc. as input features. The backend of the webpage involves SQL database and the machine learning model where the data entered for real time prediction will be stored in SQL database and the machine learning model would make predictions based upon the entered data; further the frontend of the webpage is built using HTML and CSS. The web-page could be extensively used by the bank managers to get an idea about those customers who are more likely to leave the services of the bank in future and hence, they could formulate some policies in order to retain those customers. 

<p>
<img src='https://user-images.githubusercontent.com/64635584/120081693-7c8ebf00-c0dc-11eb-97bf-95a4e7f35fd0.png'
 width='500', height='500'>
<img src='https://user-images.githubusercontent.com/64635584/120081710-929c7f80-c0dc-11eb-963b-629462ee05db.png', width='500', height='500'>
</p>


<h2>Complete Life Cycle of Project:</h2>
<h3>Statistical Techniques used:</h3>
<p>
  <ol>
      <li><strong>Statistical Analysis: </strong> Since the data does not contain any of the missing values, thus we start with Statistical Techniques which are not only used to understand relationships between the categorical and numerical data in depth by various statistical tests but also to explore the distributions and outliers present within numerical data.
    <ul>
      <li><strong>Analysis of Numerical Feature:</strong>We use histograms to inspect the distribution of data over different values. Boxplots are used to determine outliers in data as well as for understanding the distribution of data i.e. whether the data is skewed or not, or is it normally distributed. Ideally, we want a numerical feature to have Gaussian or Normal Distribution. If data is not normally distributed, we try to apply various transformation techniques such as logarithmic, square root transformation, etc. to the data so that it can be distributed approximately to Normal Distribution. Further, I have attached the distribution of 'CreditScore', one of the numerical faetures.</li>
      <img src='https://user-images.githubusercontent.com/64635584/120081904-8f55c380-c0dd-11eb-85d6-54e4f6df44e9.png', width='500', height='500'>
      <li><strong>Analysis of Categorical Feature using ANOVA Test:</strong> There are 2 statistical methods which are used to analyze categorical data with respect to each of the numerical feature i.e. T-test and One-Way ANOVA (Analysis of Variance). T-test or One-Way ANOVA is used to test whether the categories of a categorical feature with respect to a particular numerical feature have the same mean or not. If they have same mean, then it means they are not explaining or adding any variance to that particular numerical feature, hence we can conclude to drop that categorical feature since it is not adding or explaining any of the variance with respect to a particular numerical feature. Remember, T-test is used for categorical features with number of categories =2 and One-Way ANOVA is used for categorical features with number of categories greater than 2. ANOVA Result for 'Geography', a categorical feature. </li>
      <img src='https://user-images.githubusercontent.com/64635584/120081743-bc55a680-c0dc-11eb-8bdf-606b9691e9f0.png', width='500', height='600'>
    </ul></li>
    <li><strong>Data Preprocessing: </strong> Following processes are involved during data preprocessing: 
    <ul>
        <li><strong>Feature Encoding: </strong>The process of Data Preprocessing starts with Feature Encoding which simply refers to the process of converting the categorical data into discrete numerical data since machine learning algorithms can only process numerical data. the data is splitted into training and test data where training data is composed of 75% of data and rest 25% of data is used to evaluate whether the model trained using the 75% data is performing good on the instances which the model hasn’t seen during training. The attached figure shows the proportion of data for positive class and negative class in percentage (for training data it is 1540 out of 7500 instances i.e. 20.533% and for test data it is 497 out of 2500 instances i.e. 19.880%.</li>
      <img src='https://user-images.githubusercontent.com/64635584/120081920-aa283800-c0dd-11eb-9bd7-7db3aa03be79.png', width='500', height='500'>
      <li><strong>Feature Scaling: </strong> Feature Scaling transforms all the numerical features to have values within the range 0 and 1 proportionately so that the convergence rate of the algorithm gets faster and it takes less time to reach the global minimum.</li>
      <li><strong>Feature Selection: </strong> I have used 3 techniques for feature selection: 
       <ul>
        <li><strong>Correlation Matrix</strong> is observed to check if there exists multicollinearity in the data or not. Hopefully, none of the input features had high correlation with respect to each other.</li> 
        
        <li><strong>Chi-squared test of independence </strong>, where the dependence of each of the input feature is computed with respect to the target feature and a score is calculated which is displayed as bars in the image attached.</li> 
<img src='https://user-images.githubusercontent.com/64635584/120081944-c4faac80-c0dd-11eb-8c2c-4718829e46b0.png', width='500', height='500'>. 
        
        <li><strong>L1 Based Feature Selection</strong> is a technique where we use L1 regularization to regularize the parameters learnt by the classification model (Support Vector Machines) for each of the input feature and since all of the input features are scaled to have values within the same range, their weights can be used to rank their importance. The attached figure shows the weights with respect to each of the input features.</li>
           <img src='https://user-images.githubusercontent.com/64635584/120081964-df348a80-c0dd-11eb-9c2b-1558d0427a59.png', width='500', height='500'>
       </ul></li></ul>
    </li>
    <li><strong>Model Building:</strong> Out of several classification algorithms that exist in machine learning community, I have used the 5 most common ones which are (1) Logistic Regression, (2) Support Vector Classifier, (3) Decision Trees Classifier, (4) Random Forest Classifier and (5) Xtreme Gradient Boosted Trees for Classification. Hyperparameter Tuning allows us to choose the best values for the parameters of our algorithms using Cross Validation. I did the hyperparameter tuning until I observed decrement in the AUC Score or f1_score.</li>
    <li><strong>Model Evaluation:</strong>
    <ul>
      <li><strong>Precision Recall Tradeoff: </strong>Precision Recall Curve helps us getting the idea of the best precision-recall tradeoff for a particular machine learning model.</li>
      <li><strong>ROC Curve: </strong> ROC Curve is the curve obtained on plotting True Positive rate along y-axis and False Positive Rate along x-axis. It is basically used to compare different classifiers and choose the best out of it. The selection of a classifier is done based upon the Area Under th Curve of ROC.<strong> Generally for a classifier to be perform well, it should as high True Positive Rate as possible and as low False Positive Rate as possible.</strong>. Further, the figure displays Precisio-Recall Curve, ROC Curve and AUC Bar Chart for the above listed Machine Learning Algorithms and it can be observed that XGBoost Classifier has the highest AUC Score but the number of hyperparameters are a lot too more; hence, we go with Support Vector Classifier which has just 1% less AUC Score than XGBoost.</li>
       <img src='https://user-images.githubusercontent.com/64635584/120081984-fecbb300-c0dd-11eb-91f6-d907da7ae8b2.png', width='800', height='600'>
      <li><strong>Overfitting of Model: </strong> Checking whether the model generalizes for new data on which the  algorithm has not been trained on is a special requirement for the finalisez Machine Learning Algorithm. From the attached figure, it can be easily displayed that the metrics for training and validation data for SVC are almost the same which means the model is generalizing well on new data.</li>
       <img src='https://user-images.githubusercontent.com/64635584/120081764-de4f2900-c0dc-11eb-86ff-e691b9e23c39.png', width='500', height='500'>
      </ul>
    </li>
    <li><strong>Web Application</strong>A Web Application is built as discussed above where the Bank Managers would enter the details of customer and get an idea of whether the customer would leave the services of the bank in future. Thus, they can further create some policies to retain them.</li>
<li></li>
    <li><strong>SQL DataBase:</strong>The data is stored in database as can be viewed from the following figure.</li>
    <img src='https://user-images.githubusercontent.com/64635584/120081831-24a48800-c0dd-11eb-917b-d20c0b00d7d0.png', width='500', height='500'>
  </ol>
</p>

<h3>About Files:</h3>
Templates folder contains the 2 templates which of HTML and styled using CSS which gives the structure of the web page.
Static folder contains the Cascading Style Sheets with the help of which we style our webpage.
Churn_Modelling.csv is the csv file on which data is trained.
Data Science end to end Bank Churning Project is the .ipynb Jupyter Notebook which conatins full life cycle of the data science project where Statistical Techniques are also used for analyzing Categorical and Numerical Data along with the model building.
home.py is the file which contains the web application code.
model_final.pkl is the file of the finalised model.
<hr></hr>
<p>© 2021 Animesh Shukla</p>
