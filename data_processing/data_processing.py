import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import os
from typing import Tuple, Dict, List, Optional

class TransactionProcessor:
    """Process and analyze transaction data."""

    def __init__(self, file_path: str):
        """
        Initialize the processor with a file path.

        Args:
            file_path: Path to the transaction CSV file
        """
        self.file_path = file_path
        self.data = None
        self.cleaned_data = None
        self.summary_stats = {}

    def load_data(self) -> pd.DataFrame:
        """
        Load transaction data from CSV file.

        Returns:
            DataFrame containing the raw transaction data
        """
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Successfully loaded {len(self.data)} transactions.")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    def clean_data(self) -> pd.DataFrame:
        """
        Clean the transaction data by:
        - Converting dates to datetime
        - Standardizing classification tags
        - Handling missing values
        - Converting amounts to numeric values

        Returns:
            DataFrame containing the cleaned transaction data
        """
        if self.data is None:
            print("No data loaded. Please call load_data() first.")
            return pd.DataFrame()

        # Create a copy to avoid modifying the original
        cleaned = self.data.copy()

        # Convert date strings to datetime objects (format: DD/MM/YYYY)
        try:
            cleaned['Transaction_Date'] = pd.to_datetime(
                cleaned['Transaction_Date'],
                format='%d/%m/%Y',
                errors='coerce'
            )
        except Exception as e:
            print(f"Error converting dates: {e}. Will try alternative formats.")
            # Try alternative formats
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y']:
                try:
                    cleaned['Transaction_Date'] = pd.to_datetime(
                        cleaned['Transaction_Date'],
                        format=fmt,
                        errors='coerce'
                    )
                    break
                except:
                    continue

        # Add year and month columns for easier analysis
        cleaned['Year'] = cleaned['Transaction_Date'].dt.year
        cleaned['Month'] = cleaned['Transaction_Date'].dt.month
        cleaned['Day'] = cleaned['Transaction_Date'].dt.day

        # Convert transaction time to datetime.time objects
        try:
            cleaned['Transaction_Time'] = pd.to_datetime(
                cleaned['Transaction_Time'],
                format='%H:%M:%S',
                errors='coerce'
            ).dt.time
        except Exception as e:
            print(f"Error converting times: {e}")

        # Create hour of day for time analysis
        cleaned['Hour'] = pd.to_datetime(cleaned['Transaction_Time'], format='%H:%M:%S', errors='coerce').dt.hour

        # Standardize Classification_Tag (remove spaces, make lowercase)
        tag_mapping = {
            'Pre-Funding': 'Pre_Funding',
            'Pre Funding': 'Pre_Funding',
            'Pre-funding': 'Pre_Funding',
            'Prefunding': 'Pre_Funding',
            'Card_Payments': 'Card_Payments',
            'Card Payments': 'Card_Payments',
            'CardPayments': 'Card_Payments',
            'Bill_Payments': 'Bill_Payments',
            'Bill Payments': 'Bill_Payments',
            'Bill-Payments': 'Bill_Payments',
            'BillPayments': 'Bill_Payments',
            'Withdrawals': 'Withdrawals',
            'Withdrawals.  ': 'Withdrawals',
            'Withdrawals - ': 'Withdrawals',
            'Bank Charges': 'Bank_Charges'
        }

        cleaned['Classification_Tag'] = cleaned['Classification_Tag'].map(
            lambda x: tag_mapping.get(x, x)
        )

        # Ensure Amount is numeric
        cleaned['Amount'] = pd.to_numeric(cleaned['Amount'], errors='coerce')

        # Fill any NaN values with appropriate defaults (avoid inplace=True)
        cleaned['Amount'] = cleaned['Amount'].fillna(0)
        cleaned['Status'] = cleaned['Status'].fillna('Unknown')

        # Drop rows with critical missing data
        cleaned = cleaned.dropna(subset=['Transaction_ID', 'Customer_ID', 'Transaction_Date'])

        # Store the cleaned data
        self.cleaned_data = cleaned
        print(f"Data cleaning complete. {len(cleaned)} valid transactions remaining.")

        return cleaned

    def analyze_transaction_patterns(self) -> Dict:
        """
        Analyze transaction patterns like:
        - Daily/weekly patterns
        - Transaction type distribution
        - Customer behavior

        Returns:
            Dictionary containing pattern analysis results
        """
        if self.cleaned_data is None:
            print("No cleaned data available. Please call clean_data() first.")
            return {}

        data = self.cleaned_data
        patterns = {}

        # Add day of week for pattern analysis
        data['DayOfWeek'] = data['Transaction_Date'].dt.day_name()

        # Transactions by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        patterns['day_of_week'] = data['DayOfWeek'].value_counts().reindex(day_order).to_dict()

        # Transactions by tag and status
        patterns['tag_status'] = data.groupby(['Classification_Tag', 'Status']).size().to_dict()

        # Average transaction amount by tag
        patterns['avg_by_tag'] = data.groupby('Classification_Tag')['Amount'].mean().to_dict()

        # Success rate by transaction type (% completed)
        success_rate = data.groupby('Classification_Tag')['Status'].apply(
            lambda x: (x == 'Completed').sum() / len(x) * 100
        ).to_dict()
        patterns['success_rate'] = success_rate

        # Customer transaction frequencies
        customer_freq = data.groupby('Customer_ID').size()
        patterns['customer_frequency'] = {
            'single_tx': (customer_freq == 1).sum(),
            'two_to_five': ((customer_freq >= 2) & (customer_freq <= 5)).sum(),
            'six_to_ten': ((customer_freq >= 6) & (customer_freq <= 10)).sum(),
            'more_than_ten': (customer_freq > 10).sum()
        }

        # Amount distribution by hour
        patterns['hourly_amounts'] = data.groupby('Hour')['Amount'].mean().to_dict()

        # Print some insights
        print("\n===== Transaction Pattern Analysis =====")
        print("Day of Week Distribution:")
        for day, count in patterns['day_of_week'].items():
            print(f"  {day}: {count} transactions")

        print("\nSuccess Rate by Transaction Type:")
        for tag, rate in success_rate.items():
            print(f"  {tag}: {rate:.2f}% completed")

        print("\nCustomer Transaction Frequency:")
        for label, count in patterns['customer_frequency'].items():
            print(f"  {label}: {count} customers")

        return patterns

    def modify_dataset(self) -> pd.DataFrame:
        """
        Modify the dataset by:
        - Dropping unnecessary columns
        - Converting specified columns to categorical

        Returns:
            DataFrame containing the modified dataset
        """
        if self.cleaned_data is None:
            print("No cleaned data available. Please call clean_data() first.")
            return pd.DataFrame()

        # Drop the specified columns
        columns_to_drop = ['Transaction_ID', 'Customer_ID', 'Transaction_Date', 'Description', 'Transaction_Time']
        modified_data = self.cleaned_data.drop(columns=columns_to_drop)

        # Convert specified columns to categorical
        categorical_columns = ['Classification_Tag', 'Status', 'DayOfWeek']
        modified_data[categorical_columns] = modified_data[categorical_columns].astype('category')

        return modified_data

    def save_dataset(self, output_file_csv: str, output_file_parquet: str) -> None:
        """
        Save the modified dataset to CSV and Parquet files.

        Args:
            output_file_csv: Path to save the CSV file
            output_file_parquet: Path to save the Parquet file
        """
        if self.cleaned_data is None:
            print("No cleaned data available. Please call clean_data() first.")
            return

        # Modify the dataset
        modified_data = self.modify_dataset()

        # Save to CSV
        modified_data.to_csv(output_file_csv, index=False)
        print(f"Modified dataset saved to {output_file_csv}")

        # Save to Parquet
        modified_data.to_parquet(output_file_parquet)
        print(f"Modified dataset saved to {output_file_parquet}")

def main():
    """
    Main function to execute the data processing workflow.
    """
    print("Transaction Data Processing Script")
    print("=================================")

    # Input file path
    input_file = r'C:\Users\HP\Documents\credrails\data\transactions.csv'

    # Output file paths
    output_file_csv = r'C:\Users\HP\Documents\credrails\data\modified_dataset.csv'
    output_file_parquet = r'C:\Users\HP\Documents\credrails\data\modified_dataset.parquet'

    # Initialize processor
    processor = TransactionProcessor(input_file)

    # Load and clean data
    processor.load_data()
    processor.clean_data()

    # Analyze transaction patterns (to create the DayOfWeek column)
    processor.analyze_transaction_patterns()

    # Save the modified dataset
    processor.save_dataset(output_file_csv, output_file_parquet)

    print("\nData processing complete!")

if __name__ == "__main__":
    main()