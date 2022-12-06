# Python Final Project


## Problem
Seoul provides these residents with bicycles that can be useful, for example to get to work. This means of transport is effective, particularly in reducing pollution. The city must therefore ensure the availability of bicycles.

The dataset SeoulBikeData shows how many bikes are rented at a specific date and with a certain context: humidity level, temperature, visibility, season and many more. It initially contains 8760 rows and 14 columns. 

The goal is to isolate our “rented bike count” column and try to predict how many bikes will be rented at a given time. 


**Attribute Information:** \
Date : year-month-day \
Rented Bike count - Count of bikes rented at each hour \
Hour - Hour of he day \
Temperature-Temperature in Celsius \
Humidity - % \
Windspeed - m/s \
Visibility - 10m \
Dew point temperature - Celsius \
Solar radiation - MJ/m2 \
Rainfall - mm \
Snowfall - cm \
Seasons - Winter, Spring, Summer, Autumn \
Holiday - Holiday/No holiday \
Functional Day - NoFunc(Non Functional Hours), Fun(Functional hours) 


## 5 Steps
### Preprocessing

1- Renaming of columns.\
2- Deleting rows where it's a Non Functional Hour because no bikes are rented at this time.\
3- Temperature and Dew point Temperature are highly correlated so we delete this last column.\
4- Converting Date to datetime.\
5- Exploding the column "Date" to get 3 new columns: Day, Month and Year.

**Tests to see the dependance between our target value and other columns:**\
-**Pearson test** for quantitative columns: our target is dependant of all of those (p-value<5%)\
-**Anova test** for qualitative columns: our target is dependant of all of those (p-value<5%)\

### Visualization
Using of **2 librairies**:\
-Matplotlib\
-Seaborn\

The first plot is to see **number of bikes rented per season**: people rent more bikes in summer.

It's also interesting to see the **proportion of bikes rented per month and per hour**.

Finally, a **heatmap** to see the **correlation** between columns is also present. It permit to see the high correlation between temperature and the previous column named dew point temperature.

Other visualizations are in the machine learning part, like scatter plots.



### Model Training
- Encoding categorical variables like "season" with get_dummies() for our study.
- Spliting data into train and test sets (80%-20%).

### Machine Learning
In this part, we analyze the following algorithms to get the best for making predictions:
- LinearRegression()
- Ridge
- Lasso
- Support Vector Machine Regression
- DecisionTreeRegressor
- BaggingRegressor
- RandomForestRegressor\

For each model, we calculate score of the model, R² score and mean squared error. There are also scatter plots to visualize if the prediction is correct.

### Results
After that study, we get best scores for Random Forest Regressor model. \
We obtain:\
**score train:**  98.08323764062936 %\
**score test:**  85.27606812026087 %\
**r² score:** 85.27606812026087 %\
**root mean squared error:** 240.42493334537656\
and a **cross score validation** of 86.14802299742188 %.

So this model is more than acceptable to make predictions.
