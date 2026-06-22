# am planning to use the meta's prophet model for this

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

df = pd.read_csv(r'D:\ocally assignment\business_daily.csv')

class ProphetForecaster:
    def __init__(self , df):
        self.df = df.copy()
        self.models = {}
        self.results = {}

    def _prepare_prophet_df(self, biz_df):
        biz_df = biz_df.sort_values("date")

        # renaming date to ds and revenue to y
        prophet_df = biz_df[["date", "revenue"]].rename(columns={"date" : "ds", "revenue" : "y"})

        # adding regressors
        prophet_df["visits"] = biz_df["visits"].values
        prophet_df["marketing_spend"] = biz_df["marketing_spend"].values
        prophet_df["is_weekend"] = biz_df["is_weekend"].values
        prophet_df["local_event_flag"] = biz_df["local_event_flag"].values

        return prophet_df

    def train(self, business_id):
        biz_df = self.df[self.df["business_id"] == business_id].copy()
        if len(biz_df) < 30:
            print(f"amount of rows too low -> skipping")
            return None

        prophet_df = self._prepare_prophet_df(biz_df)

        # initialize prophet model
        model = Prophet(
            yearly_seasonality= True,
            weekly_seasonality= True,
            daily_seasonality= True,
            seasonality_mode= "multiplicative",
            changepoint_prior_scale= 0.05, # regulariztion
        )

        model.add_regressor("visits", prior_scale=10.0)
        model.add_regressor("marketing_spend", prior_scale= 5.0)
        model.add_regressor("is_weekend" , prior_scale=2.0)
        model.add_regressor("local_event_flag", prior_scale=3.0)

        model.fit(prophet_df)
        self.models[business_id] = model
        self.results[business_id] = prophet_df

        return model

    
    def predict_14d(self, business_id, future_marketing=None):
        if business_id not in self.models:
            self.train(business_id)
            
        model = self.models[business_id]
        prophet_df = self.results[business_id]
        
        # Create future dataframe for 14 days
        last_date = prophet_df['ds'].max()
        future = model.make_future_dataframe(periods=14, include_history=False)
        
        # We need to provide regressor values for the future
       
        last_visits = prophet_df['visits'].tail(30).mean()
        last_weekend = 1 if future['ds'].iloc[0].dayofweek >= 5 else 0
        last_event = 0  
        last_marketing = 0  
        
        # Populate regressors for future
        future['visits'] = last_visits
        future['marketing_spend'] = last_marketing
        future['is_weekend'] = future['ds'].apply(lambda x: 1 if x.dayofweek >= 5 else 0)
        future['local_event_flag'] = last_event

        if future_marketing:
            mask = future['ds'] == future_marketing['date']
            future.loc[mask, 'marketing_spend'] = future_marketing['amount']
        
        forecast = model.predict(future)

        result = forecast[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'revenue'})
        result['revenue'] = result['revenue'].clip(lower=0)  # No negative revenue
        
        return result
    
    def evaluate(self, test_days=14):
        results = []
        
        for biz in self.df['business_id'].unique():
            biz_df = self.df[self.df['business_id'] == biz].sort_values('date')
            
            if len(biz_df) < 30:
                continue

            train_df = biz_df.iloc[:-test_days].copy()
            test_df = biz_df.iloc[-test_days:].copy()

            temp_prophet = ProphetForecaster(train_df)
            temp_prophet.train(biz)
            
            # Make predictions day by day (walk-forward) or all at once
            predictions = []
            history = train_df.copy()
            
            for i, row in test_df.iterrows():
                # Predict next day
                pred_df = temp_prophet.predict_14d(biz)
                next_pred = pred_df.iloc[0]['revenue']
                predictions.append(next_pred)
                
                new_row = pd.DataFrame([{
                    'business_id': biz,
                    'date': row['date'],
                    'revenue': row['revenue'],
                    'visits': row['visits'],
                    'marketing_spend': row['marketing_spend'],
                    'is_weekend': row['is_weekend'],
                    'local_event_flag': row['local_event_flag']
                }])
                history = pd.concat([history, new_row], ignore_index=True)

                temp_prophet = ProphetForecaster(history)
                temp_prophet.train(biz)

            actuals = test_df['revenue'].values
            mape = mean_absolute_percentage_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            
            results.append({
                'business': biz,
                'model': 'prophet',
                'mape': mape,
                'mae': mae,
                'n_test': len(test_df)
            })
        
        return pd.DataFrame(results)


def estimate_marketing_impact(df, business_id, spend_amount=200):
    biz_df = df[df['business_id'] == business_id].copy()
    mkt_days = biz_df[biz_df['marketing_spend'] > 0]
    non_mkt_days = biz_df[biz_df['marketing_spend'] == 0]

    if len(mkt_days) < 3 or len(non_mkt_days) < 10:
        return {'confidence': 'low', 'roi': None, 'recommendation': 'insufficient_data'}

    # Control for day‑of‑week
    avg_mkt_by_day = mkt_days.groupby(mkt_days['date'].dt.dayofweek)['revenue'].mean()
    avg_non_by_day = non_mkt_days.groupby(non_mkt_days['date'].dt.dayofweek)['revenue'].mean()
    lifts = []
    for day, avg_mkt in avg_mkt_by_day.items():
        avg_non = avg_non_by_day.get(day, np.nan)
        if pd.notna(avg_non) and avg_non > 0:
            lifts.append((avg_mkt - avg_non) / avg_non)

    if not lifts:
        return {'confidence': 'low', 'roi': None, 'recommendation': 'insufficient_data'}

    avg_lift = np.mean(lifts)
    recent_avg_revenue = biz_df['revenue'].tail(30).mean()
    expected_boost = recent_avg_revenue * avg_lift
    roi = expected_boost / spend_amount

    confidence = 'high' if len(lifts) >= 4 and np.std(lifts) < 0.3 else 'medium'

    return {
        'confidence': confidence,
        'roi': roi,
        'expected_boost': expected_boost,
        'recommendation': 'spend' if roi > 1.0 else 'hold'
    }


def main():
    print("=" * 60)
    print("PROPHET 14 DAY REVENUE FORECAST & MARKETING RECOMMENDATION")
    print("=" * 60)

    # Load data
    df = pd.read_csv('business_daily.csv')
    df['date'] = pd.to_datetime(df['date'])
    print(f"\nLoaded {len(df):,} records from {df['date'].min().date()} to {df['date'].max().date()}")

    # Initialize forecaster
    forecaster = ProphetForecaster(df)

    # Train models for all businesses
    print("\n[1/3] Training Prophet models...")
    for biz in df['business_id'].unique():
        forecaster.train(biz)
    print("Training complete.")

    # Generate 14-day forecasts
    print("\n[2/3] Generating 14-day forecasts...")
    forecasts = {}
    for biz in df['business_id'].unique():
        if biz not in forecaster.models:
            continue
        fcast = forecaster.predict_14d(biz)
        forecasts[biz] = fcast
        total = fcast['revenue'].sum()
        print(f"  {biz}: ${total:,.2f} total | avg ${fcast['revenue'].mean():,.2f}/day")

    # Marketing recommendation
    print("\n[3/3] Marketing spend recommendation (Should we spend $200 this week?)")
    best_biz = None
    best_roi = -float('inf')
    best_info = None

    for biz in df['business_id'].unique():
        impact = estimate_marketing_impact(df, biz, spend_amount=200)
        if impact['confidence'] != 'low' and impact['roi'] is not None:
            if impact['roi'] > best_roi:
                best_roi = impact['roi']
                best_biz = biz
                best_info = impact

    if best_biz:
        print(f"\nRECOMMENDATION: Spend $200 on **{best_biz}**")
        print(f"   Expected ROI: {best_roi:.2f}x")
        print(f"   Expected revenue boost: ${best_info['expected_boost']:,.2f}")
        print(f"   Confidence: {best_info['confidence']}")
        # Find the best day to spend (highest forecasted revenue day)
        best_day_forecast = forecasts[best_biz]
        best_day = best_day_forecast.loc[best_day_forecast['revenue'].idxmax(), 'date']
        print(f"   Best day to spend: {best_day.date()}")
    else:
        print("\nNot enough marketing data across any business for a reliable recommendation.")
        print("   Recommendation: Hold this week, collect more data.")

    # Save forecasts
    print("\nSaving forecasts to prophet_forecast.csv ...")
    all_forecasts = pd.concat([f.assign(business_id=biz) for biz, f in forecasts.items()], ignore_index=True)
    all_forecasts.to_csv('prophet_forecast.csv', index=False)
    print("Done.")

if __name__ == "__main__":
    main()

