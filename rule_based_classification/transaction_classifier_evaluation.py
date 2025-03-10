import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os

# Import the classifier from the previous script
# Note: Ensure the file is in the same directory or update the import
from transaction_classifier import TransactionStatusClassifier


class ClassifierEvaluator:
    """
    A class to evaluate the performance of the transaction status classifier.
    """
    
    def __init__(self, classifier):
        """
        Initialize the evaluator with a classifier.
        
        Args:
            classifier: An instance of TransactionStatusClassifier
        """
        self.classifier = classifier
        self.status_categories = ['Completed', 'Pending', 'Cancelled', 'Refunded', 'Chargeback']
    
    def evaluate(self, transactions_df: pd.DataFrame, true_label_col: str = 'Status', 
                pred_label_col: str = 'Classified_Status') -> Dict:
        """
        Evaluate the classifier using various metrics.
        
        Args:
            transactions_df: DataFrame containing transaction data with true and predicted labels
            true_label_col: Column name for true labels
            pred_label_col: Column name for predicted labels
            
        Returns:
            Dict: Dictionary containing various evaluation metrics
        """
        # Extract true and predicted labels
        y_true = transactions_df[true_label_col].values
        y_pred = transactions_df[pred_label_col].values
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate per-class metrics
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0, 
                                         labels=self.status_categories)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0,
                                           labels=self.status_categories)
        
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0,
                                   labels=self.status_categories)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0,
                                     labels=self.status_categories)
        
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0,
                           labels=self.status_categories)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0,
                             labels=self.status_categories)
        
        # Get detailed classification report
        class_report = classification_report(y_true, y_pred, target_names=self.status_categories, 
                                            output_dict=True, zero_division=0)
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred, labels=self.status_categories)
        
        # Compile metrics
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro, 
            'recall_weighted': recall_weighted,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix
        }
        
        return metrics
    
    def calculate_class_distribution(self, transactions_df: pd.DataFrame, 
                                   label_col: str) -> pd.Series:
        """
        Calculate the distribution of classes.
        
        Args:
            transactions_df: DataFrame containing transaction data
            label_col: Column name for labels
            
        Returns:
            pd.Series: Series containing class distribution
        """
        return transactions_df[label_col].value_counts(normalize=True)
    
    def calculate_error_analysis(self, transactions_df: pd.DataFrame, 
                               true_label_col: str = 'Status',
                               pred_label_col: str = 'Classified_Status') -> pd.DataFrame:
        """
        Perform error analysis to understand misclassifications.
        
        Args:
            transactions_df: DataFrame containing transaction data with true and predicted labels
            true_label_col: Column name for true labels
            pred_label_col: Column name for predicted labels
            
        Returns:
            pd.DataFrame: DataFrame with error analysis
        """
        # Get misclassifications
        errors_df = transactions_df[transactions_df[true_label_col] != transactions_df[pred_label_col]].copy()
        
        # Add error type column
        errors_df['Error_Type'] = errors_df.apply(
            lambda row: f"{row[true_label_col]} → {row[pred_label_col]}", axis=1
        )
        
        # Group by error type
        error_counts = errors_df['Error_Type'].value_counts().reset_index()
        error_counts.columns = ['Error_Type', 'Count']
        
        # Add percentage
        total_errors = error_counts['Count'].sum()
        error_counts['Percentage'] = (error_counts['Count'] / total_errors * 100).round(2)
        
        return error_counts
    
    def plot_confusion_matrix(self, conf_matrix: np.ndarray, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            conf_matrix: Confusion matrix
            figsize: Figure size
            
        Returns:
            plt.Figure: Confusion matrix figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.status_categories,
                   yticklabels=self.status_categories,
                   ax=ax)
        
        # Set labels
        plt.xlabel('Predicted Status')
        plt.ylabel('True Status')
        plt.title('Confusion Matrix')
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_metrics_by_class(self, class_report: Dict, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot precision, recall, and F1-score by class.
        
        Args:
            class_report: Classification report dictionary
            figsize: Figure size
            
        Returns:
            plt.Figure: Metrics by class figure
        """
        # Extract class metrics
        metrics_df = pd.DataFrame()
        
        for status in self.status_categories:
            if status in class_report:
                metrics_df[status] = [
                    class_report[status]['precision'],
                    class_report[status]['recall'],
                    class_report[status]['f1-score']
                ]
        
        metrics_df.index = ['Precision', 'Recall', 'F1-Score']
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        metrics_df.plot(kind='bar', ax=ax)
        
        # Set labels
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('Performance Metrics by Status Category')
        plt.legend(title='Status')
        plt.ylim(0, 1)
        
        # Add value labels
        for i, container in enumerate(ax.containers):
            ax.bar_label(container, fmt='%.2f', padding=3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def evaluate_by_segment(self, transactions_df: pd.DataFrame, 
                          segment_col: str,
                          true_label_col: str = 'Status',
                          pred_label_col: str = 'Classified_Status') -> pd.DataFrame:
        """
        Evaluate classifier performance by segments.
        
        Args:
            transactions_df: DataFrame containing transaction data
            segment_col: Column to segment by (e.g., 'Classification_Tag')
            true_label_col: Column name for true labels
            pred_label_col: Column name for predicted labels
            
        Returns:
            pd.DataFrame: DataFrame with metrics by segment
        """
        segment_metrics = []
        
        # Loop through each segment
        for segment, segment_df in transactions_df.groupby(segment_col):
            if len(segment_df) < 10:  # Skip small segments
                continue
                
            # Calculate metrics for this segment
            accuracy = accuracy_score(segment_df[true_label_col], segment_df[pred_label_col])
            f1 = f1_score(segment_df[true_label_col], segment_df[pred_label_col], 
                         average='weighted', zero_division=0)
            
            # Record metrics
            segment_metrics.append({
                'Segment': segment,
                'Count': len(segment_df),
                'Percentage': len(segment_df) / len(transactions_df) * 100,
                'Accuracy': accuracy,
                'F1_Score': f1
            })
        
        # Convert to DataFrame
        segment_metrics_df = pd.DataFrame(segment_metrics)
        
        # Sort by count
        if not segment_metrics_df.empty:
            segment_metrics_df = segment_metrics_df.sort_values('Count', ascending=False)
        
        return segment_metrics_df
    
    def evaluate_by_amount_range(self, transactions_df: pd.DataFrame,
                               amount_col: str = 'Amount',
                               true_label_col: str = 'Status',
                               pred_label_col: str = 'Classified_Status') -> pd.DataFrame:
        """
        Evaluate classifier performance by amount ranges.
        
        Args:
            transactions_df: DataFrame containing transaction data
            amount_col: Column with transaction amounts
            true_label_col: Column name for true labels
            pred_label_col: Column name for predicted labels
            
        Returns:
            pd.DataFrame: DataFrame with metrics by amount range
        """
        # Define amount ranges
        def get_amount_range(amount):
            if amount < 0:
                return "Negative"
            elif amount < 100:
                return "0-100"
            elif amount < 500:
                return "100-500"
            elif amount < 1000:
                return "500-1000"
            else:
                return "1000+"
        
        # Add amount range column
        transactions_df = transactions_df.copy()
        transactions_df['AmountRange'] = transactions_df[amount_col].apply(get_amount_range)
        
        # Evaluate by amount range
        return self.evaluate_by_segment(transactions_df, 'AmountRange', 
                                       true_label_col, pred_label_col)


def run_evaluation():
    """Run a comprehensive evaluation of the transaction status classifier."""
    try:
        # Load transaction data
        transactions_df = pd.read_csv(r'C:\Users\HP\Documents\credrails\data\transactions.csv')
        print(f"Loaded {len(transactions_df)} transactions for evaluation")
        
        # Initialize classifier and evaluator
        classifier = TransactionStatusClassifier()
        evaluator = ClassifierEvaluator(classifier)
        
        # Enhance data with additional flags
        enhanced_df = classifier.enhance_transaction_data(transactions_df)
        
        # Classify transactions
        result_df = classifier.classify_transactions(enhanced_df)
        
        # Calculate overall metrics
        metrics = evaluator.evaluate(result_df)
        
        # Print overall metrics
        print("\n=== OVERALL PERFORMANCE METRICS ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
        
        # Print classification report
        print("\n=== CLASSIFICATION REPORT ===")
        print(classification_report(result_df['Status'], result_df['Classified_Status'], 
                                   target_names=evaluator.status_categories, zero_division=0))
        
        # Calculate error analysis
        error_analysis = evaluator.calculate_error_analysis(result_df)
        print("\n=== ERROR ANALYSIS ===")
        print(error_analysis)
        
        # Calculate metrics by transaction type
        segment_metrics = evaluator.evaluate_by_segment(result_df, 'Classification_Tag')
        print("\n=== PERFORMANCE BY TRANSACTION TYPE ===")
        print(segment_metrics.to_string(index=False))
        
        # Calculate metrics by amount range
        amount_metrics = evaluator.evaluate_by_amount_range(result_df)
        print("\n=== PERFORMANCE BY AMOUNT RANGE ===")
        print(amount_metrics.to_string(index=False))
        
        # Create output directory for visualizations
        output_dir = r'C:\Users\HP\Documents\credrails\rule_based\evaluation_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot confusion matrix
        conf_matrix_fig = evaluator.plot_confusion_matrix(metrics['confusion_matrix'])
        conf_matrix_fig.savefig(f"{output_dir}/confusion_matrix.png")
        
        # Plot metrics by class
        metrics_fig = evaluator.plot_metrics_by_class(metrics['classification_report'])
        metrics_fig.savefig(f"{output_dir}/metrics_by_class.png")
        
        # Save results to CSV
        error_analysis.to_csv(f"{output_dir}/error_analysis.csv", index=False)
        segment_metrics.to_csv(f"{output_dir}/segment_metrics.csv", index=False)
        amount_metrics.to_csv(f"{output_dir}/amount_metrics.csv", index=False)
        
        # Save full results with predictions
        result_df.to_csv(f"{output_dir}/full_classification_results.csv", index=False)
        
        print(f"\nEvaluation complete. Results saved to '{output_dir}' directory")
        
        return {
            'metrics': metrics,
            'error_analysis': error_analysis,
            'segment_metrics': segment_metrics,
            'amount_metrics': amount_metrics
        }
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_evaluation()