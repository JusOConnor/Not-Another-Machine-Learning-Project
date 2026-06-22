# Not Another Machine Learning Project
I wanted to take a different approach to this project.  Yes, this IS another ML project, but I want to show how it's working, how the models evolove over time, and do it in a manner that doesn't require a Masters Degree to understand.  The idea isn't just to give users another project to run blindly, but to demystify the concepts.

***This is going to be focused on creating ML models based on stock performance.  In no way should this be your basis for financial decisions***

*That being said, IF you do make some money off this project, please feel free to send some my way.  I could work with a typical Hollywood 12%*

___
**Step 1:**  
The dataset will be built utilizing the Yahoo Finance API to pull the OHLCV.
- **O**pen - price at market open
- **H**igh - highest price of the day
- **L**ow - lowest price of the day
- **C**lose - price at market close
- **V**olume - number of shares traded.  

This data is exported to the prices_%.csv file.
```
Date,Open,High,Low,Close,Volume
2018-03-14,244.51890227060835,244.7037296005574,241.75519513417626,242.30967712402344,105895100
2018-03-15,242.82021999592808,243.4627240370959,241.54396844860167,242.04566955566406,83433000
```
The features_%.csv is built from the OHLCV data and is used by the models
```
Date,close_to_ma5,close_to_ma20,close_to_ma50,daily_return,momentum_5,momentum_20,volatility_20,volume_ratio,hl_range,target
2018-03-14,0.9949333230815984,1.0076111889018402,1.0034964886042976,-0.00513140216141994,0.00923821169931549,0.03496228808944668,0.009651743293826244,0.9998483162752482,0.012168455265086147,0
2018-03-15,0.9932031657616973,1.0055178397652138,1.0019480599895365,-0.0010895461192177436,0.003283644053667656,0.020067597213516697,0.00926176096319149,0.801885473189126,0.007927246093749977,1
```

___
**Step 2:**  
Models will be trained using three of the most popular models for tabular data applying them to each company.  We're going to keep the typical 80/20 rule where we train on 80% of the historical data and hold back the remainign 20%.  The 20% is never seen by the model and is reserved for the evalutation in Step 3.
- [Random Forest](https://en.wikipedia.org/wiki/Random_forest) - builds hundreds of simple decision trees and takes a majority vote. Good at finding nonlinear patterns without overfitting.
- [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression) - draws a straight line through the data to separate outcomes. Simple, fast, and a useful baseline to beat.
- [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) - builds trees sequentially, each one correcting the mistakes of the last. Often the strongest performer but most sensitive to the data it was trained on.

___
**Step 3:**  
Models will be compared against each other to see which one performed best.  This also the point where we start to see that there is not a "one model fits all" for the stock market, nor will the same model always be better than the others for the same stock.  This is also why we're logging hte results over time in results_log_%.csv so you can if the winning models changes across runs.

___
**Step 4:**  
Here's where we start to answer the "now what?" portion of the ML project.  The "best" models will be used to predict what's going to happen.  You can think of this as Step 2, but utilizing new data.  For example, if your training data runs through mid-2023, the model learned from that period and now you are passing in late 2023, 2024, and 2025 data it has never seen.

___
**Step 5:**  
This step is effectively steps 1 through 4, but adjusts the "end date" to reflect the current date.  There is also a safeguard in place that compares the models against the previous run.  If there is no improved the model is not updated.  The improvement level can be adjusted by updating the 'min_improvement' variable in the Ticker_Config.py file.  It is recommended to run the first 4 steps manualy to get an understand of what each step is doing and then run step 5 on a regular cadence for your go forward.  Optionally, you can skip the manual run of each step and just work with Step 5 for both the initial loads and the go-forward.
Suggested schedule:  
- Monthly - safe default
- Weekly - reasonable for more active tracking
- Daily - generally overkill  

It's worth noting models can degrade with frequent runs.  Think of it like a weather forecaster who learned all the weather patterns and climate of one city. If they switch to a new city, the same patterns may not apply. The market works similarly, patterns that predicted direction in 2020 may be meaningless in 2025.

___
**Step 6:**  
Up until this point the outputs have been model files and log entries, useful for the pipeline but not easy to interpret. This step produces a side-by-side comparison of what each model predicted versus what actually happened and produces a CSV file/s that we can import into something like Excel or PowerBI to visualize.  Alternately you could build the visuals in Jupyter.

___
**Step 7:**  
Think of this step as sort of a visualizer for this Machine Learning project.  Steps 3 through 5 are all centered around the "Now" and how the models have performed over time.  This step simulates the process historically to show what happens to the models over time as more data is being introduced.  Rather than one training run, it performs multiple sequential training and evaluation cycles on progressively larger slices of historical data. Each cycle represents what a real retraining run would have looked like at that point in time.

A few things to watch out for in the output:
1. A model that consistently improves as the training window grows suggests it is finding genuine patterns in the data rather than memorizing noise
2. A model that peaks early then declines is likely sensitive to a specific market regime that existed in the earlier data but faded over time
3. All models tracking near 50% accuracy — as you will likely see with stock data — is not a failure of the project. It is an honest result that reflects how difficult market prediction genuinely is, and it is far more valuable than an inflated score produced by a flawed evaluation method

The outputs from this step are designed to feed directly into Power BI or Excel. The detail file gives you one row per model per window, which is ideal for line charts comparing all three models across time. The summary file gives you one row per window showing which model won that cycle and whether it improved over the previous one, which works well as a table or scorecard visual.


___
## Folder Structure
```
stocks/
├── data/
├── models/
│   ├── backup/
├── reports/
│   ├── backtesting/
│   └── walkforward/
├── .gitignore
├── package.json
└── README.md
```

Run this block of code to remove all the generated files and "start over" with the project.
```
import Empty_Folders
Empty_Folders.fClearProjectFolders()
```

___

## Start Here  
I've included two Jupyter notebooks for the data loading.  
Data Management Single - Use this if you're working with a single stock ticker  
Data Management Multi - Use this to work from a list

### Python Files
```
Ticker_Config.py        # creates the variables needed to run each function
Empty_Folders.py         # deletes all the files generated by the project.  This is useful for when you want to start over
Step_1_Data.py          # pulls data, builds features.csv
Step_2_Train.py         # trains 3 models, saves .pkl files
Step_3_Compare.py       # scores all models, writes results log
Step_4_Predict.py       # loads best model, predicts today
Step_5_Retrain.py       # runs steps 1 through 4 with a default 'end_date' of today()
Step_6_Backtsting.py    # builds a dataset comapring the model predictions versus actuals
Step_7_Walkforward.py   # automatically simulates the models evolution over time
```

#### Loading Data  
For the first run I would suggest running Steps 1 through 4 to get a sense of how everything works.
```
# Import Functions
import Ticker_Config as tc
import Step_1_Data as Step1
import Step_2_Train as Step2
import Step_3_Compare as Step3
import Step_4_Predict as Step4

ticker = 'SPY'
start_date = '2018-01-01'
end_date = '2025-12-31'

cfg = tc.MLConfig(ticker, start_date, end_date)
Step1.fMain(cfg)    # replace 'Step1' with whichever step you need
```
The 'cfg' variable returned from MLConfig contains all the variables needed for each function so you won't need to worry about identifying them manually.  
Alternatively, you can just run Step_5_Retrain.py and load it all in one go.
```
import Step_5_Retrain as Step5

cfg_list = 'SPY'
start_date = '2018-01-01'
end_date = '2025-12-31'

cfg = tc.MLConfig(cfg_list, start_date, end_date)
Step5.fMain(cfg)
```

The two previosuly mentioned Jupyter notebooks have these steps prebuilt.  

#### Backtesting and Walkforward - Optional Analytics  
Both of these have two functions you will use.  The fMain() performs the analysis on individual Tickers.  The combine_backtest_files() and combine_walkforward_files() functions combine the results from each.  The combine functions are optional and make importing the data into programs like PowerBI and Excel easier since it combines the results into single files for import rather than having to link each Ticker's output individually.
___


#### File Flow  

>**Step 1:**  
>data\features_%.csv  
>data\prices_%.csv  
>
>**Step 2:**  
>data\test_set_%.csv  
>models\gradient_boosting_%.pkl  
>models\logistic_regression_%.pkl  
>models\random_forest_%.pkl  
>
>**Step 3:**  
>models\best_model_%.txt  
>reports\results_log_%.csv  
>
>**Step 4:**  
>*no files created in this step*  
>
>**Step 5:**  
>**see steps 1 through 3*  
>
>**Step 6:**  
>reports\backtesting\backtest_%.csv  
>reports\backtesting\backtest_summary_%.csv  
>reports\backtesting\combined_detail.csv  
>reports\backtesting\combined_summary.csv  
>
>**Step 7:**  
>reports\walkforward\walkforward_%.csv
>reports\walkforward\walkforward_summary_%.csv
>reports\walkforward\combined_detail.csv  
>reports\walkforward\combined_summary.csv  

___
## Results Samples  
### Backtesting_Analytics.ipynb

**Overall Model Accuracy**  
<img src="Screenshots/BT Overall Accuracy by Model.png" width="30%">  

**Montly Accuracy by Model**  
<img src="Screenshots/BT Monthly Model Accuracy.png" width="30%"> 

**Predictions vs Actuals Over Time**  
<img src="Screenshots/BT Predictions vs Actuals.png" width="60%">  

**Monthly Accuracy by Model**  
<img src="Screenshots/BT Monthly Accuracy by Model.png" width="60%">  

**Overall Accuracy - Multi**  
<img src="Screenshots/BT Overall Accuracy - All.png" width="30%">  

**Best Model per Ticker**  
<img src="Screenshots/BT Best Model by Ticker.png" width="30%">  

**Accuracy by Model - Mult**  
<img src="Screenshots/BT Overall Accuracy by Model and Ticker.png" width="60%">  

___
### Walkforward_Analytics.ipynb

**Scores by Window and Model**  
<img src="Screenshots/WF Scores by Window and Model.png" width="30%">  

**Best Model by Window**  
<img src="Screenshots/WF Best Model by Window.png" width="30%">  

**ROC AUC and Accuracy Across Windows**  
<img src="Screenshots/WF ROC AUC and Accuracy Across Windows.png" width="30%">  

**Prediction Bias Across Windows**  
<img src="Screenshots/WF Prediction Bias Across Windows.png" width="30%">  

**Best Model per Window**  
<img src="Screenshots/WF Best Model per Window - Multi.png" width="30%">  

**Which Model Wins Most Often**  
<img src="Screenshots/WF Which Model Wins Most Often - Multi.png" width="60%">  

___
## Repurposing for different data  
**Step 1**  
This where most of the work happens. The entire engineer_features function is domain-specific. Moving averages, momentum, and volume ratios are stock concepts. For something like customer sales data you would replace those with sales domain features like rolling average order value, days since last purchase, purchase frequency, or month-over-month growth rate. The target variable also changes, instead of "will price go up tomorrow" it might be "will this customer churn" or "will they buy again this month."  

**Step 2**  
FEATURE_COLUMNS then just needs to reflect whatever columns step 1 produces. It is a manifest of what step 1 creates, so they always move together.  

**Steps 3 through 7**  
Do not require changes at all. Scoring, comparison, prediction, backtesting, and retraining are all domain-agnostic. They just see numbers going in and predictions coming out.  

**MLConfig**  
Would need updated default dates or none at all if the data comes from a database rather than a date range pull.