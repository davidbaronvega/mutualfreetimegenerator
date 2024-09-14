#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
file_path = '/Users/david/Downloads/DBV.LR.xlsx'
df_new = pd.read_excel(file_path)

# Define the days of the week
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def parse_availability(row):
    """Parse each respondent's availability from a row of the DataFrame."""
    availability = {time_slot: {day: 0 for day in days_of_week} for time_slot in df_new.columns[2:-1]}
    for time_slot in df_new.columns[2:-1]:
        time_slot_availability = row[time_slot]
        if pd.notna(time_slot_availability):
            available_days = time_slot_availability.split(', ')
            for day in available_days:
                if day in availability[time_slot]:
                    availability[time_slot][day] = 1
    return availability

# Aggregate availability for each time slot across all respondents
total_availability = {time_slot: {day: 0 for day in days_of_week} for time_slot in df_new.columns[2:-1]}

for _, row in df_new.iterrows():
    individual_availability = parse_availability(row)
    for time_slot, days in individual_availability.items():
        for day, available in days.items():
            if available:
                total_availability[time_slot][day] += 1

# Convert aggregated availability into a DataFrame for easier analysis
availability_df = pd.DataFrame(total_availability).T  # Transpose to have time slots as rows

# Filter to find time slots where at least 2 people are available
filtered_availability = availability_df[availability_df >= 2].dropna(how='all')

# Assuming the rest of your code remains the same and leads up to the plotting section

# Visualization
plt.figure(figsize=(15, 7))

# Create an offset for each time slot to avoid overlapping bars
time_slots = filtered_availability.index.tolist()
days_indices = {day: i for i, day in enumerate(days_of_week)}
n_bars = len(time_slots)
bar_width = 0.1  # Adjust as necessary for your number of time slots
offsets = np.linspace(-bar_width*n_bars/2, bar_width*n_bars/2, n_bars)

for offset, (time_slot, row) in zip(offsets, filtered_availability.iterrows()):
    days = [days_indices[day] + offset for day in row.dropna().index]
    counts = row.dropna().values
    plt.bar(days, counts, width=bar_width, label=time_slot)

plt.title('Number of People Available per Time Slot and Day')
plt.xlabel('Day of the Week')
plt.ylabel('Number of People Available')

# Set x-ticks to be in the middle of the groups
plt.xticks(range(len(days_of_week)), days_of_week, rotation=45)
plt.legend(title="Time Slots", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[ ]:




