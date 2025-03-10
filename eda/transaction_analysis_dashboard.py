import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def transaction_analysis_dashboard(csv_path):
    """
    Analyze transaction data and generate visualizations similar to the React dashboard.
    
    Parameters:
    csv_path (str): Path to the CSV file containing transaction data
    """
    # Load data
    print("Loading transaction data...")
    df = pd.read_csv(csv_path)
    
    # Set up the matplotlib figure layout
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(20, 25))
    
    # Status Distribution (Pie Chart)
    print("Generating status distribution chart...")
    plt.subplot(3, 2, 1)
    status_counts = df['Status'].value_counts()
    plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', 
            startangle=90, colors=sns.color_palette('bright', len(status_counts)))
    plt.title('Transaction Status Distribution')
    plt.axis('equal')
    
    # Classification Distribution (Bar Chart)
    print("Generating classification distribution chart...")
    plt.subplot(3, 2, 2)
    
    # Normalize classification tags (similar to React code)
    def normalize_classification(tag):
        if pd.isna(tag):
            return "Unknown"
        tag = str(tag)
        if "Pre" in tag and "Fund" in tag:
            return "Pre-Funding"
        elif "Transfer" in tag:
            return "Transfers"
        elif "Withdraw" in tag:
            return "Withdrawals"
        elif "Card" in tag:
            return "Card_Payments"
        elif "Bill" in tag:
            return "Bill_Payments"
        elif "Bank" in tag:
            return "Bank Charges"
        else:
            return "Other"
    
    df['Normalized_Classification'] = df['Classification_Tag'].apply(normalize_classification)
    classification_counts = df['Normalized_Classification'].value_counts()
    
    sns.barplot(x=classification_counts.index, y=classification_counts.values)
    plt.xticks(rotation=45, ha='right')
    plt.title('Transaction Classification Distribution')
    plt.tight_layout()
    
    # Hourly Transaction Volume
    print("Generating hourly transaction volume chart...")
    plt.subplot(3, 2, 3)
    
    # Extract hour from Transaction_Time
    df['Hour'] = df['Transaction_Time'].str.split(':', expand=True)[0].astype(int)
    hourly_counts = df['Hour'].value_counts().sort_index()
    
    # Fill in missing hours with 0
    hourly_data = pd.Series(0, index=range(24))
    hourly_data.update(hourly_counts)
    
    sns.barplot(x=hourly_data.index, y=hourly_data.values)
    plt.xticks(range(24), [f"{h:02d}" for h in range(24)], rotation=45)
    plt.title('Hourly Transaction Volume')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Transactions')
    
    # Average Amount by Category
    print("Generating average transaction amount chart...")
    plt.subplot(3, 2, 4)
    
    avg_amount = df.groupby('Normalized_Classification')['Amount'].mean().round(2)
    sns.barplot(x=avg_amount.index, y=avg_amount.values)
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Transaction Amount by Category')
    plt.ylabel('Average Amount ($)')
    
    # Daily Transaction Volume Trend
    print("Generating daily transaction trend chart...")
    plt.subplot(3, 2, 5)
    
    # Convert date strings to datetime
    df['Date'] = pd.to_datetime(df['Transaction_Date'], format='%d/%m/%Y')
    
    # Get the last 30 days of data
    daily_counts = df.groupby('Date').size()
    last_30_days = daily_counts.sort_index().tail(30)
    
    plt.plot(last_30_days.index, last_30_days.values, marker='o')
    plt.gcf().autofmt_xdate()
    plt.title('Daily Transaction Volume (Recent 30 Days)')
    plt.ylabel('Number of Transactions')
    
    # Add key insights text panel
    plt.subplot(3, 2, 6)
    plt.axis('off')
    completed_pct = df[df['Status'] == 'Completed'].shape[0] / df.shape[0] * 100
    chargeback_pct = df[df['Status'] == 'Chargeback'].shape[0] / df.shape[0] * 100
    
    # Calculate potential fraud indicators (outliers in amount)
    Q1 = df['Amount'].quantile(0.25)
    Q3 = df['Amount'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_pct = df[df['Amount'] > (Q3 + 1.5 * IQR)].shape[0] / df.shape[0] * 100
    
    insights = (
        "Key Insights:\n\n"
        f"• Completed Transactions: Approximately {completed_pct:.1f}% of all transactions are completed successfully\n\n"
        f"• Transaction Categories: {df['Normalized_Classification'].value_counts().index[0]} and "
        f"{df['Normalized_Classification'].value_counts().index[1]} make up the largest portion of transactions\n\n"
        "• Transaction Timing: Transaction volume varies throughout the day with notable patterns\n\n"
        f"• Transaction Amounts: {avg_amount.idxmax()} has the highest average transaction amount (${avg_amount.max():.2f})\n\n"
        f"• Potential Fraud Indicators: About {outlier_pct:.1f}% of transactions have unusually high amounts\n\n"
        f"• Chargebacks: Approximately {chargeback_pct:.1f}% of transactions result in chargebacks"
    )
    plt.text(0.1, 0.5, insights, fontsize=12, verticalalignment='center')
    
    plt.tight_layout(pad=3.0)
    plt.savefig('transaction_analysis.png', dpi=150, bbox_inches='tight')
    print("Analysis complete! Saved to 'transaction_analysis.png'")
    
    # Create interactive dashboard with Plotly (optional)
    create_interactive_dashboard(df)
    
def create_interactive_dashboard(df):
    """
    Create an interactive dashboard using Plotly
    
    Parameters:
    df (DataFrame): The transaction data
    """
    print("Generating interactive Plotly dashboard...")
    
    # Create a subplot figure
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Transaction Status Distribution', 
            'Transaction Classification Distribution',
            'Hourly Transaction Volume', 
            'Average Transaction Amount by Category',
            'Daily Transaction Volume (Recent 30 Days)', 
            'Key Insights'
        ),
        specs=[
            [{"type": "pie"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "table"}]
        ]
    )
    
    # Status Distribution (Pie Chart)
    status_counts = df['Status'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=status_counts.index, 
            values=status_counts.values,
            textinfo='percent+label'
        ),
        row=1, col=1
    )
    
    # Classification Distribution (Bar Chart)
    classification_counts = df['Normalized_Classification'].value_counts()
    fig.add_trace(
        go.Bar(
            x=classification_counts.index, 
            y=classification_counts.values,
            marker_color='rgb(158,202,225)'
        ),
        row=1, col=2
    )
    
    # Hourly Transaction Volume
    hourly_counts = df['Hour'].value_counts().sort_index()
    hourly_data = pd.Series(0, index=range(24))
    hourly_data.update(hourly_counts)
    
    fig.add_trace(
        go.Bar(
            x=[f"{h:02d}" for h in hourly_data.index], 
            y=hourly_data.values,
            marker_color='rgb(142,124,195)'
        ),
        row=2, col=1
    )
    
    # Average Amount by Category
    avg_amount = df.groupby('Normalized_Classification')['Amount'].mean().round(2)
    fig.add_trace(
        go.Bar(
            x=avg_amount.index, 
            y=avg_amount.values,
            marker_color='rgb(106,168,79)'
        ),
        row=2, col=2
    )
    
    # Daily Transaction Volume Trend
    daily_counts = df.groupby('Date').size()
    last_30_days = daily_counts.sort_index().tail(30)
    
    fig.add_trace(
        go.Scatter(
            x=last_30_days.index, 
            y=last_30_days.values,
            mode='lines+markers',
            marker=dict(size=8),
            line=dict(width=3)
        ),
        row=3, col=1
    )
    
    # Key Insights table
    completed_pct = df[df['Status'] == 'Completed'].shape[0] / df.shape[0] * 100
    chargeback_pct = df[df['Status'] == 'Chargeback'].shape[0] / df.shape[0] * 100
    
    # Calculate potential fraud indicators (outliers in amount)
    Q1 = df['Amount'].quantile(0.25)
    Q3 = df['Amount'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_pct = df[df['Amount'] > (Q3 + 1.5 * IQR)].shape[0] / df.shape[0] * 100
    
    insights = [
        ["Completed Transactions", f"Approximately {completed_pct:.1f}% of all transactions are completed successfully"],
        ["Transaction Categories", f"{df['Normalized_Classification'].value_counts().index[0]} and {df['Normalized_Classification'].value_counts().index[1]} make up the largest portion"],
        ["Transaction Timing", "Transaction volume varies throughout the day with notable patterns"],
        ["Transaction Amounts", f"{avg_amount.idxmax()} has the highest average transaction amount (${avg_amount.max():.2f})"],
        ["Potential Fraud", f"About {outlier_pct:.1f}% of transactions have unusually high amounts"],
        ["Chargebacks", f"Approximately {chargeback_pct:.1f}% of transactions result in chargebacks"]
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Metric", "Insight"],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=list(zip(*insights)),
                fill_color='lavender',
                align='left'
            )
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1200, 
        width=1000,
        title_text="Financial Transaction Analysis Dashboard",
        showlegend=False
    )
    
    # Save interactive dashboard
    fig.write_html("transaction_dashboard_interactive.html")
    print("Interactive dashboard saved to 'transaction_dashboard_interactive.html'")

if __name__ == "__main__":
    # Replace with your actual CSV path
    csv_path = r"C:\Users\HP\Documents\credrails\data\transactions.csv"
    transaction_analysis_dashboard(csv_path)