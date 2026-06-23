
## getting insights on the data
the first obvious thing i did when i got the task was to find insights on the data. I started with **performing EDA** on the data in order to understand the data better. Here are my key insights: the shape of the data was `(3028 , 9)` and there were no null values. The data had `6` unique businesses namely `blue_bottle_cafe`, `'corner_bakery'`, `'evergreen_boutique'`, `'iron_pulse_gym'`, `'nonna_trattoria'`, `'shear_genius_salon'`
The range of dates was between `2024-07-01` to `2025-12-31` and the total number of days were `549`. There should have been a total of `3249` rows (549 * 6) but since we had only `3028` rows that means that our dataset had missing values.

My next step was to check which businesses reported less days (reason for missing rows). I found that `corner_bakery` had only reported 400 days out of 549 which means 149 days were missing. Similarly `iron_pulse_gym` had only reported 432 days meaning 117 days were missing. Next i checked the **revenue distribution per business** . `nonna_trattoria` has the max skew (1.49) and is heavily right tailed distribution. `Corner_bakery` had the highest skew at the lowest revenue which means outliers were heavily dominating this. `Evergreen boutique` has nearly symmetric skew.

Next i checked **weekly seasonality**. Every business peaks friday/saturday without exception. `Iron pulse gym` has higher on monday compared to other days. Next was checking **the marketing spends**. `Corner bakery` spends only on 2 days so it can be ignored. `Evergreen Botique` shows -2% lift which might be a result on spending money on bad days.

## model selection
after all these insights i had two major models in mind that i can use on this data. One was **Meta's Prophet model** and other was **XGBoost**. I decided to move forward with the prophet model specifically because of these reasons :
- it handles missing data natively whereas xgboost required imputation
- it has built in weekly seasonality, xgboost requires manual feature engineering 
- has built in US holidays
- interprets trends 

## building the model
![](Images/Pasted%20image%2020260623024056.png)
put the `daily_seasonality` to False because there's no hourly data in the dataset. Each row represents one full day. Adding regressors to our helps it in predicting the targets better. 

One thing with **Prophet** is that it wont work if we dont rename the data and target column to ds and y respectively. Also it doesnot work if a specific number of business has less than 30 rows. The priority scale here acts as a confidence meter meaning the higher the `prior_scale` the higher will be the impact. 

I got **38% mean absolute percentage error** which is quiet high but our specific dataset had sparse marketing data and missing values. Our data has inherent volatility. Example `nonna_tratttoria` had average revenue of  $6075 per day but for some days the revenue was $20000+. Also our marketing data is sparse. It had only 30 odd marketing days out of 549 (6% of total)  which means that our model had very few examples to learn from. One way we could improve accuracy is if we reduce outlier impact by capping extreme values, another way is tuning model's hyperparameter  per business.

**the current flow of my model looks like** :
*train prophet models --> generate 14 days forecast -->  recommend if we spend on marketing  or not*

my model predicts spending 200$ on `nonna_trattoria` this week. expected roi would be `8.68x` and expected revenue boost will be $1736. Best day to spend is `2026-01-13`. At the end we save all the forecasts to another csv file.