#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 12:38:39 2025

@author: ravneetkaursaini
"""




#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nycflights13 import flights
import plotly.express as px
import sys
print(sys.executable)



#load the dataset
flights_df=flights.copy()
#create a copy for safe usage

#take a quick look at the data
print(flights_df.head())
print(flights_df.info())

# (a) arrival delay of 2 or more hours
arr_delay_2hr=flights_df[flights_df['arr_delay']>=120]

#(b) Flights to Houston(IAH or HOU)
houston_flights = flights_df[flights_df['dest'].isin(['IAH', 'HOU'])]
                           
#(C) flights operated by american, united or delta
selected_carriers=flights_df[flights_df['carrier'].isin(['AA','UA','DA'])]

#(D) flights departed in summer (July, August, September)
summer_flights = flights_df[flights_df['month'].isin([7, 8, 9])]


#(e) arrived more than 2 hours late but did not depart late
arr_2hr_no_dep = flights_df[(flights_df['arr_delay'] > 120) & (flights_df['dep_delay'] <= 0)]


#(f) delayed by at least 1 hour but made up over 30 mintues in flight
made_up_time=flights_df[(flights_df['dep_delay']>=60) & (flights_df['dep_delay']-flights_df['arr_delay'])>30]

#(g) departed between midnight and 6am (inclusive)
early_morning=flights_df[(flights_df['dep_time']>=0) & (flights_df['dep_time']<=600)]
                          

#(a) most delayed flights (arrvial delay)
most_delayed=flights_df.sort_values(by='arr_delay',ascending=False)

#(b) flights that departed the earliest 
earliest_departure=flights_df.sort_values(by='dep_time',ascending=True)

#Gain in air: departure delay - arrival delay
flights_df['gain_in_air']=flights_df['dep_delay']-flights_df['arr_delay']

#positive gain means the flight made up time in the air 
print(flights_df[['dep_delay','arr_delay','gain_in_air']].head())



#speed=distance/air time + 60 to get miles per hour
flights_df['speed_mph']=flights_df['distance']/flights_df['air_time']*60

#fastest flights
fastest_flights=flights_df.sort_values(by='speed_mph',ascending=False)

#longest and shortest flight by distance
longest_flights=flights_df.sort_values(by='distance',ascending=False)
shortest_flights=flights_df.sort_values(by='distance',ascending=True)

                                       
avg_arr_delay=flights_df.groupby('carrier')['arr_delay'].mean().sort_values(ascending=False)
print(avg_arr_delay)

#airline with the highest arrival delay 
print("Airline with highest average arrival delay:",avg_arr_delay.idxmax())





# === Dashboard Functions ===

def get_delay_chart():
    fig, ax = plt.subplots(figsize=(8, 5))
    avg_arr_delay.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title('Average Arrival Delay by Airline')
    ax.set_xlabel('Airline')
    ax.set_ylabel('Average Arrival Delay (Minutes)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xticklabels(avg_arr_delay.index, rotation=0)
    return fig

def get_top_delayed_flights(n=10):
    top_flights = flights_df.sort_values(by='arr_delay', ascending=False).head(n)
    return top_flights[['carrier', 'flight', 'origin', 'dest', 'arr_delay']]


def get_summary_metrics():
    return {
        "worst_airline": avg_arr_delay.idxmax(),
        "worst_delay": avg_arr_delay.max(),
        "fastest_flight_speed": fastest_flights.iloc[0]['speed_mph'],
        "longest_flight_distance": longest_flights.iloc[0]['distance'],
        "most_delayed_flight": most_delayed.iloc[0][['carrier', 'flight', 'arr_delay']].to_dict()
    }





