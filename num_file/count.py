import os
import pandas as pd
from datetime import datetime, timedelta

# Define the directory path where the CSV files are located
directory = './raw_file/'

# Iterate through each CSV file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Convert the 'timestamp' column to datetime
        df['timestamp'] = pd.to_datetime(df['dateut'], unit='s')
        df['timestamp'] += timedelta(hours=9)
        
        # Extract the date and hour from the 'timestamp' column
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        
        # Group by date, hour, and count the number of bdaddr occurrences
        hourly_counts = df.groupby(['date', 'hour'])['bdaddr'].count().reset_index()
        
        output_filename = os.path.splitext(filename)[0]
        print(output_filename)
        output_path = os.path.join('./num_file/', output_filename+'.csv')
        hourly_counts.to_csv(output_path, index=False)