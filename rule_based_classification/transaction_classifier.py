import pandas as pd
import numpy as np
import datetime
import re
from typing import Dict, List, Union, Optional

class TransactionStatusClassifier:
    """
    A rule-based classifier for transaction statuses.
    
    This classifier implements the rules defined in the transaction status classification system
    to determine the appropriate status for each transaction based on its characteristics.
    """
    
    def __init__(self):
        """Initialize the classifier with default thresholds and settings."""
        # Configuration parameters
        self.high_value_threshold = 1000
        self.pending_review_days = 3
        self.refund_chargeback_window_days = 90
        self.dispute_keywords = [
            'dispute', 'fraud', 'unauthorized', 'not recognized', 'chargeback'
        ]
        self.refund_keywords = [
            'refund', 'return', 'reversal', 'money back', 'cancelled order'
        ]
        
        # Status transition matrix (from_status → to_status)
        self.allowed_transitions = {
            'Completed': {'Refunded', 'Chargeback'},
            'Pending': {'Completed', 'Cancelled'},
            'Cancelled': set(),  # Terminal state
            'Refunded': {'Chargeback'},
            'Chargeback': {'Refunded'}
        }
        
    def classify_transaction(self, transaction: Dict) -> str:
        """
        Determine the status of a transaction based on its characteristics.
        
        Args:
            transaction: A dictionary containing transaction data
            
        Returns:
            str: The classified status
        """
        # Apply rules in hierarchical order
        if self._matches_chargeback_rules(transaction):
            return "Chargeback"
        elif self._matches_refunded_rules(transaction):
            return "Refunded"
        elif self._matches_cancelled_rules(transaction):
            return "Cancelled"
        elif self._matches_pending_rules(transaction):
            return "Pending"
        else:
            return "Completed"
    
    def _matches_chargeback_rules(self, transaction: Dict) -> bool:
        """Check if transaction matches any chargeback rules."""
        # Check for dispute keywords in description
        if transaction.get('Description') and any(keyword in transaction['Description'].lower() 
                                                for keyword in self.dispute_keywords):
            return True
            
        # High-value transaction with any dispute indicators
        if (transaction.get('Amount', 0) >= self.high_value_threshold and 
            (transaction.get('Disputed') == True or 
             transaction.get('FraudFlag') == True)):
            return True
            
        # Transactions escalated to card issuer
        if transaction.get('EscalatedToIssuer') == True:
            return True
            
        return False
    
    def _matches_refunded_rules(self, transaction: Dict) -> bool:
        """Check if transaction matches any refunded rules."""
        # Check for refund keywords in description
        if transaction.get('Description') and any(keyword in transaction['Description'].lower() 
                                                for keyword in self.refund_keywords):
            return True
            
        # Duplicate transaction
        if transaction.get('IsDuplicate') == True:
            return True
            
        # Transaction classified as error
        if transaction.get('ErrorFlag') == True:
            return True
            
        # Failed delivery
        if transaction.get('DeliveryStatus') == 'Failed':
            return True
            
        return False
    
    def _matches_cancelled_rules(self, transaction: Dict) -> bool:
        """Check if transaction matches any cancelled rules."""
        # Customer or merchant cancellation
        if transaction.get('CancellationRequested') == True:
            return True
            
        # Insufficient funds
        if transaction.get('InsufficientFunds') == True:
            return True
            
        # Authorization timeout
        if transaction.get('AuthorizationTimeout') == True:
            return True
            
        # Payment method rejected
        if transaction.get('PaymentMethodRejected') == True:
            return True
            
        # Security verification failure
        if transaction.get('SecurityVerificationFailed') == True:
            return True
            
        return False
    
    def _matches_pending_rules(self, transaction: Dict) -> bool:
        """Check if transaction matches any pending rules."""
        # Transaction initiated but not settled
        if transaction.get('SettlementStatus') == 'Not Settled':
            return True
            
        # Awaiting verification
        if transaction.get('VerificationStatus') == 'Pending':
            return True
            
        # Processing delay
        if transaction.get('ProcessingDelay') == True:
            return True
            
        # Awaiting batch processing
        if self._is_after_hours(transaction.get('Transaction_Time')):
            return True
            
        # Manual review flagged
        if transaction.get('ManualReviewFlag') == True:
            return True
            
        # Risk assessment hold
        if transaction.get('RiskAssessmentHold') == True:
            return True
            
        # High-value transactions with additional verification
        if (transaction.get('Amount', 0) >= self.high_value_threshold and 
            transaction.get('VerificationStatus') != 'Completed'):
            return True
            
        return False
    
    def _is_after_hours(self, transaction_time: Optional[str]) -> bool:
        """Check if transaction occurs after business hours."""
        if not transaction_time:
            return False
            
        try:
            # Parse the transaction time
            hour = int(transaction_time.split(':')[0])
            # After business hours: 6PM - 6AM
            return hour >= 18 or hour < 6
        except (ValueError, IndexError):
            return False
    
    def is_valid_transition(self, current_status: str, new_status: str) -> bool:
        """
        Check if a status transition is valid according to the transition matrix.
        
        Args:
            current_status: The current transaction status
            new_status: The proposed new transaction status
            
        Returns:
            bool: True if the transition is allowed, False otherwise
        """
        if current_status == new_status:
            return True
            
        return new_status in self.allowed_transitions.get(current_status, set())
    
    def classify_transactions(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify all transactions in a DataFrame.
        
        Args:
            transactions_df: DataFrame containing transaction data
            
        Returns:
            pd.DataFrame: The input DataFrame with an additional 'Classified_Status' column
        """
        result_df = transactions_df.copy()
        
        # Apply classification to each row
        result_df['Classified_Status'] = result_df.apply(
            lambda row: self.classify_transaction(row.to_dict()), axis=1
        )
        
        return result_df
    
    def enhance_transaction_data(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance transaction data with additional flags needed for classification.
        
        This method adds synthetic flags based on existing data to simulate
        a real-world scenario where these flags might come from other systems.
        
        Args:
            transactions_df: DataFrame containing transaction data
            
        Returns:
            pd.DataFrame: Enhanced DataFrame with additional classification flags
        """
        enhanced_df = transactions_df.copy()
        
        # Add dispute-related flags
        enhanced_df['Disputed'] = enhanced_df['Description'].str.contains('|'.join(self.dispute_keywords), 
                                                                       case=False, 
                                                                       regex=True, 
                                                                       na=False)
        
        # Simulate fraud flags (random for demonstration)
        np.random.seed(42)  # For reproducibility
        enhanced_df['FraudFlag'] = np.random.choice(
            [True, False], 
            size=len(enhanced_df), 
            p=[0.02, 0.98]  # 2% fraud rate
        )
        
        # Mark duplicate transactions (same amount, customer, close timestamps)
        enhanced_df['IsDuplicate'] = False
        
        # Group by Customer_ID and Amount to find potential duplicates
        for _, group in enhanced_df.groupby(['Customer_ID', 'Amount']):
            if len(group) > 1:
                # Mark transactions on same day as duplicates
                same_day_groups = group.groupby('Transaction_Date')
                for _, day_group in same_day_groups:
                    if len(day_group) > 1:
                        # Get the indices of duplicates (skip the first one)
                        duplicate_indices = day_group.index[1:]
                        enhanced_df.loc[duplicate_indices, 'IsDuplicate'] = True
        
        # Add verification status
        enhanced_df['VerificationStatus'] = np.random.choice(
            ['Completed', 'Pending', 'Failed'],
            size=len(enhanced_df),
            p=[0.85, 0.10, 0.05]
        )
        
        # Add settlement status
        enhanced_df['SettlementStatus'] = 'Settled'
        enhanced_df.loc[enhanced_df['Status'] == 'Pending', 'SettlementStatus'] = 'Not Settled'
        
        # Add high-value transaction flag
        enhanced_df['HighValue'] = enhanced_df['Amount'] >= self.high_value_threshold
        
        # Add manual review flag for certain transactions
        enhanced_df['ManualReviewFlag'] = False
        enhanced_df.loc[enhanced_df['HighValue'] & 
                       (enhanced_df['Classification_Tag'] == 'Pre-Funding'), 'ManualReviewFlag'] = True
        
        return enhanced_df


def main():
    """Main function to demonstrate the transaction status classifier."""
    try:
        # Load transaction data
        transactions_df = pd.read_csv(r'C:\Users\HP\Documents\credrails\data\transactions.csv')
        print(f"Loaded {len(transactions_df)} transactions")
        
        # Initialize classifier
        classifier = TransactionStatusClassifier()
        
        # Enhance data with additional flags
        enhanced_df = classifier.enhance_transaction_data(transactions_df)
        
        # Classify transactions
        result_df = classifier.classify_transactions(enhanced_df)
        
        # Compare original status with classified status
        result_df['Status_Match'] = result_df['Status'] == result_df['Classified_Status']
        match_percentage = result_df['Status_Match'].mean() * 100
        
        print(f"\nStatus Classification Results:")
        print(f"Match percentage: {match_percentage:.2f}%")
        
        # Show distribution of statuses
        print("\nOriginal Status Distribution:")
        print(result_df['Status'].value_counts(normalize=True).mul(100).round(1))
        
        print("\nClassified Status Distribution:")
        print(result_df['Classified_Status'].value_counts(normalize=True).mul(100).round(1))
        
        # Show some examples of mismatches
        mismatches = result_df[~result_df['Status_Match']].head(10)
        if not mismatches.empty:
            print("\nSample Status Mismatches:")
            for _, row in mismatches.iterrows():
                print(f"Transaction {row['Transaction_ID']}: Original={row['Status']}, " 
                     f"Classified={row['Classified_Status']}, Amount=${row['Amount']:.2f}")
        
        # Save results to CSV
        output_file = r'C:\Users\HP\Documents\credrails\rule_based\transaction_status_results.csv'
        result_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()