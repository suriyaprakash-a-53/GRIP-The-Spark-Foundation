# Student Score Prediction Model

This project implements a **Student Score Prediction Model** using Python, aimed at predicting students' scores based on various input features. The model leverages machine learning algorithms to assist educators in identifying areas where students may need improvement.

## Overview :-

The **Student Score Prediction Model** uses historical student data to train a machine learning model capable of predicting students' scores. It aids in data-driven decision-making for enhancing educational strategies.

# Title :- Student Score Prediction Model

# Created by :- Suriyaprakash A

Key functionalities include:-
- Cleaning and preprocessing the dataset.
- Training a predictive model using Python libraries.
- Visualizing the results and performance metrics.

---

## Features :-

- Predict student scores based on historical data.
- Intuitive data visualizations using Matplotlib and Seaborn.
- Easy-to-use interface for model interaction.
- Supports generative AI tools like GitHub Copilot for code suggestions.

---

## Technologies Used :-

The project is built using the following technologies:

- **Programming Language**: Python
- **Frameworks**: Flask (optional for deployment)
- **Libraries**:
  - Pandas
  - NumPy
  - Scikit-learn
  - Matplotlib
  - Seaborn
- **Generative AI Tools**: GitHub Copilot, ChatGPT (for code debugging)

---

## Setup and Installation :-

1. **Clone the Repository**:
   
   git clone https://github.com/suriyaprakash-a-53/student-score-prediction.git
   cd student-score-prediction
   

2. **Set up a Virtual Environment**:
   
   python -m venv env
   source env/bin/activate  # For Windows: env\Scripts\activate
   

3. **Install Dependencies**:
   
   pip install -r requirements.txt
   

4. **Run the Application**:
   
   python main.py
   

## Code :-

#Import the needed libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
     

# reading the given data set
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data is Readed Sucessfully")

data.head(7)
     
Data is Readed Sucessfully
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
5	1.5	20
6	9.2	88

# Cheacking null values
data.isnull().sum()
     
0
Hours	0
Scores	0

dtype: int64
There is no null values so,Data cleaning is not Required.


# Visualize our data's
# Scatterplot
# sns.set_style('darkgrid')

plt.scatter(x= data['Hours'],y= data['Scores'])
plt.title('Hours vs percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage scored')
plt.show()
print(data.corr())
     

           Hours    Scores
Hours   1.000000  0.976191
Scores  0.976191  1.000000

# Regressionplot

sns.set_style('darkgrid')
sns.regplot(x= data['Hours'],y= data['Scores'])
plt.title('Hours vs percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage scored')
plt.show()
print(data.corr())
     

           Hours    Scores
Hours   1.000000  0.976191
Scores  0.976191  1.000000
The variables are positively correlated.

Training Model
Split the Given data

# Defining x and y from the Data
x = data.iloc[:,:-1].values
y = data.iloc[:,1].values

# spliting the Data in 2(train and test)
train_x,test_x,train_y,test_y = train_test_split(x,y,random_state=0)
     
** " Fitting the Data into model " **

# Using Linear Regression

regression = LinearRegression()
regression.fit(train_x,train_y)

print("*****Model is Successfully Trained*****")

     
*****Model is Successfully Trained*****

# Predicting the Percntage of Marks

pred_y = regression.predict(test_x)
prediction = pd.DataFrame({'Hours': [i[0] for i in test_x],'Predicted Scores':[k for k in pred_y]})
prediction
     
Hours	Predicted Scores
0	1.5	16.844722
1	3.2	33.745575
2	7.4	75.500624
3	2.5	26.786400
4	5.9	60.588106
5	3.8	39.710582
6	1.9	20.821393
Compare the Original and Predicted Scores

compare_scores =pd.DataFrame({'Original Scores': test_y, 'Predicted Scores':pred_y})
compare_scores
     
Original Scores	Predicted Scores
0	20	16.844722
1	27	33.745575
2	69	75.500624
3	30	26.786400
4	62	60.588106
5	35	39.710582
6	24	20.821393
Visual Comparision between Original and Predicted Scores

# Scatterplot
plt.scatter(x= test_x,y= test_y)
plt.plot(test_x,pred_y)
plt.title('Original vs Predicted')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage scored')
plt.show()
print(data.corr())
     

           Hours    Scores
Hours   1.000000  0.976191
Scores  0.976191  1.000000
Evaluating the Model

# accuracy calculating of model
print('mean absolute error:',mean_absolute_error(test_y,pred_y))
     
mean absolute error: 4.130879918502482
Predicted score of 9.25 hours

hours =[9.25]
sol = regression.predict([hours])
print('scores={}',format(round(sol[0],3)))
     
scores={} 93.893
Training Scores vs Tested Scores

print("Train : ",regression.score(train_x,train_y)*100)
print("Test : ",regression.score(test_x,test_y)*100)
     
Train :  94.84509249326872
Test :  93.67661043365057


---![st (1)](https://github.com/user-attachments/assets/6bde41dc-7e3c-48f0-b3b6-a766044974c6)


## How to Use :-

1. **Input Dataset**: Upload a dataset containing student details (e.g., hours studied, test preparation, etc.).
2. **Train the Model**: Train the predictive model using the preprocessed dataset.
3. **Make Predictions**: Use the model to predict student scores.
4. **Visualize Results**: View data insights and model performance metrics.


## Project Highlights :-

- **Data Analysis**: Conducted exploratory data analysis (EDA) to understand dataset trends.
- **Model Development**: Utilized machine learning algorithms like Linear Regression for predictions.
- **Generative AI Tools**: Leveraged GitHub Copilot and ChatGPT to expedite development and debug complex code efficiently.

---

## Future Enhancements :-

- Add support for multiple machine learning models for improved prediction accuracy.
- Deploy the model using Flask or a cloud service like AWS or Azure.
- Develop a user-friendly web interface for educators to interact with the model.
- Include additional features like student demographics and performance history.

---

Feel free to customize this file further based on your project's specific requirements or additional features!
