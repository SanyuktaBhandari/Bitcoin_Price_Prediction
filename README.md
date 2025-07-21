# BITCOIN PRICE PREDICTION USING MACHINE LEARNING


# Introduction: 

Bitcoin is a decentralized digital currency that has gained global attention due to its high volatility and potential for substantial returns.<br/> This project aims to analyse historical Bitcoin data and use machine learning algorithms to predict whether the price will increase or decrease on the following day.
<br/>

# Objective:

To develop a predictive model using historical Open, High, Low, and Close (OHLC) Bitcoin price data that forecasts whether the closing price of the next day will be higher than the current day's.


# Dataset Description:

•	The dataset contains historical OHLC data of Bitcoin from 2014 to 2022  <br/>
•	Features include: Date, Open, High, Low, Close, Adjusted Close (removed after redundancy check), and Volume.
<br/>

# Data Preprocessing and Cleaning:

•	Checked for missing values: none found.<br/>
•	Removed the 'Adj Close' column as it duplicated 'Close' values.<br/>
•	Converted the 'Date' column to datetime format and extracted additional features: year, month, and day.<br/>
•	Added a binary feature is_quarter_end to indicate whether a month marks the end of a financial quarter.
<br/>

# Exploratory Data Analysis (EDA):

EDA is an approach to analyzing the data using visual techniques. It is used to discover trends, and patterns, or to check assumptions with the help of statistical summaries and graphical representations.<br/>

•	Line Plot:<br/>
o	The 'Close' price was plotted over time to observe the overall trend and volatility.<br/>
o	Insight: Bitcoin exhibits strong uptrends in some years (e.g., 2017, 2020) and sharp corrections.

•	Distribution Plots (Histograms):<br/>
o	Open, High, Low, Close prices were visualized.<br/>
o	Insight: Price values are right-skewed, showing many lower prices and fewer extremely high values.

•	Box Plots:<br/>
o	Identified outliers and range for each price feature.<br/>
o	Insight: High volatility is evident through a wide range and frequent outliers.

•	Yearly Bar Charts:<br/>
o	Average Open, High, Low, and Close prices grouped by year.<br/>
o	Insight: Clear year-on-year growth until 2021, a huge price explosion in the year 2021, followed by a dip in 2022.
<br/>

# Feature Engineering:

Feature Engineering helps to derive some valuable features from the existing ones. These extra features sometimes help in increasing the performance of the model significantly and certainly help to gain deeper insights into the data.

•	open-close: Captures intra-day movement.<br/>
•	low-high: Represents volatility within the day.<br/>
•	is_quarter_end: Affects market behaviour due to end of financial quarters.<br/>
•	target: A binary variable indicating whether the next day’s closing price is higher (1) or not (0).<br/><br/>

Target Distribution:<br/>
•	Plotted as a pie chart.<br/>
•	Insight: The dataset is relatively balanced between price increases and decreases, making it suitable for binary classification.<br/><br/>

Correlation Analysis:<br/>
•	Correlation matrix and heatmap plotted.<br/>
•	Insight: Strong correlations (>0.9) were present among OHLC values, but engineered features provided distinct signals.
<br/><br/>

# Model Preparation:

•	Features standardized using StandardScaler.<br/>
•	Manual time-based split used: 70% for training, 30% for testing.<br/>
This manual split keeps the time order — important for time series problems like stock prices

•	Models used:<br/>
o	Logistic Regression: Simple and interpretable model for binary classification.<br/>
o	Support Vector Machine (Polynomial Kernel): Support Vector Machine with a polynomial kernel (can model more complex decision boundaries)<br/>
o	XGBoost Classifier: A powerful, tree-based model that often performs best for structured data.


# Model Evaluation:

•	Metric used: ROC AUC Score<br/>
This is because instead of predicting the hard probability that is 0 or 1, we would like it to predict soft probabilities that are continuous values between 0 to 1.<br/> 
And with soft probabilities, the ROC-AUC curve is generally used to measure the accuracy of the predictions.


•	Performance summary:

o	Logistic Regression:<br/>
	Training Accuracy: 0.5351<br/>
	Validation Accuracy: 0.5171

o	Support Vector Machine (Polynomial):<br/>
	Training Accuracy: 0.4621<br/>
	Validation Accuracy: 0.4876

o	XGBClassifier:<br/>
	Training Accuracy: 0.9994<br/>
	Validation Accuracy: 0.5329<br/>


•	Insight: <br/>
XGBClassifier has the highest performance but it is prone to overfitting as the difference between the training and the validation accuracy is too high. <br/>
But in the case of the Logistic Regression, this is not the case.


# Confusion Matrix Analysis:

•	Plotted for Logistic Regression.<br/>
•	Insight: Model captures both classes but has room for improvement in precision/recall balance.
<br/>


# Conclusion and Inference:

•	Feature engineering significantly improved model performance over raw OHLC values.<br/>
•	The model predicts the direction of next-day price movement with moderate success.<br/>
•	We can observe that the accuracy achieved by the state-of-the-art ML model is no better than simply guessing with a probability of 50%. <br/>
Possible reasons for this may be the lack of data or using a very simple model to perform such a complex task as Stock Market prediction.
<br/>

# Future Enhancements:

•	Include temporal models like LSTM to capture sequential dependencies.<br/>
•	Introduce sentiment analysis from news or social media.<br/>
•	Incorporate external macroeconomic indicators.<br/>
•	Deploy the model using a dashboard interface (e.g., Streamlit).
<br/>

# Tools and Libraries Used:

•	Python, Pandas, NumPy, Matplotlib, Seaborn<br/>
•	Scikit-learn, XGBoost<br/>
•	Jupyter Notebook for development
<br/>

# Model Deployment Potential:

•	Can serve as a signal generator in algorithmic trading.<br/>
•	Basis for more advanced time-series forecasting pipelines.
  
