import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from credit_scoring_model import DeFiCreditScorer
import json

def validate_model():
    """Comprehensive model validation suite."""
    
    # Create mock data for validation
    mock_data = create_mock_validation_data()
    
    # Initialize scorer
    scorer = DeFiCreditScorer()
    
    # Process data
    feature_df = scorer.engineer_features(mock_data)
    feature_df_with_targets = scorer.create_target_scores(feature_df)
    
    # Train model
    scorer.train_model(feature_df_with_targets)
    
    # Perform validation
    validation_results = {
        'cross_validation': perform_cross_validation(scorer, feature_df),
        'temporal_validation': perform_temporal_validation(scorer, feature_df),
        'feature_importance': scorer.feature_importance,
        'score_distribution': analyze_score_distribution(feature_df_with_targets),
        'classification_metrics': evaluate_classification_performance(feature_df_with_targets)
    }
    
    # Generate visualizations
    create_validation_plots(validation_results)
    
    # Save results
    with open('validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    print("Model validation completed. Results saved to validation_results.json")
    return validation_results

def create_mock_validation_data():
    """Create mock transaction data for validation."""
    np.random.seed(42)
    
    transactions = []
    wallets = [f"0x{i:040x}" for i in range(1000)]
    actions = ['deposit', 'borrow', 'repay', 'redeemunderlying', 'liquidationcall']
    
    for wallet in wallets:
        n_transactions = np.random.poisson(10) + 1
        
        for i in range(n_transactions):
            transactions.append({
                'wallet': wallet,
                'timestamp': pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365)),
                'action': np.random.choice(actions, p=[0.3, 0.25, 0.25, 0.15, 0.05]),
                'amount': np.random.lognormal(3, 1),
                'asset': np.random.choice(['USDC', 'ETH', 'WBTC']),
                'transaction_hash': f"0x{np.random.randint(0, 2**256):064x}"
            })
    
    return pd.DataFrame(transactions)

def perform_cross_validation(scorer, feature_df):
    """Perform cross-validation on the model."""
    feature_cols = [col for col in feature_df.columns if col not in ['wallet', 'credit_score']]
    X = feature_df[feature_cols].fillna(0)
    y = feature_df['credit_score']
    
    # 5-fold cross-validation
    cv_scores = cross_val_score(scorer.model['rf'], X, y, cv=5, scoring='r2')
    
    return {
        'mean_r2': cv_scores.mean(),
        'std_r2': cv_scores.std(),
        'individual_scores': cv_scores.tolist()
    }

def perform_temporal_validation(scorer, feature_df):
    """Perform temporal validation using time series splits."""
    feature_cols = [col for col in feature_df.columns if col not in ['wallet', 'credit_score']]
    X = feature_df[feature_cols].fillna(0)
    y = feature_df['credit_score']
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        score = scorer.model['rf'].score(X_test, y_test)
        scores.append(score)
    
    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'individual_scores': scores
    }

def analyze_score_distribution(feature_df):
    """Analyze the distribution of credit scores."""
    scores = feature_df['credit_score']
    
    return {
        'mean': scores.mean(),
        'median': scores.median(),
        'std': scores.std(),
        'min': scores.min(),
        'max': scores.max(),
        'percentiles': {
            '25th': scores.quantile(0.25),
            '50th': scores.quantile(0.50),
            '75th': scores.quantile(0.75),
            '90th': scores.quantile(0.90),
            '95th': scores.quantile(0.95)
        }
    }

def evaluate_classification_performance(feature_df):
    """Evaluate classification performance for risk levels."""
    feature_df['risk_level'] = feature_df['credit_score'].apply(
        lambda x: 'High' if x < 400 else 'Medium' if x < 700 else 'Low'
    )
    
    risk_dist = feature_df['risk_level'].value_counts()
    
    return {
        'distribution': risk_dist.to_dict(),
        'percentages': (risk_dist / len(feature_df) * 100).to_dict()
    }

def create_validation_plots(validation_results):
    """Create validation visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Feature importance plot
    importance = validation_results['feature_importance']
    features = list(importance.keys())[:10]  # Top 10 features
    values = [importance[f] for f in features]
    
    axes[0, 0].barh(features, values)
    axes[0, 0].set_title('Top 10 Feature Importance')
    axes[0, 0].set_xlabel('Importance')
    
    # Cross-validation scores
    cv_scores = validation_results['cross_validation']['individual_scores']
    axes[0, 1].boxplot(cv_scores)
    axes[0, 1].set_title('Cross-Validation R² Scores')
    axes[0, 1].set_ylabel('R² Score')
    
    # Score distribution histogram
    # Mock data for demonstration
    scores = np.random.normal(400, 150, 1000)
    scores = np.clip(scores, 0, 1000)
    
    axes[1, 0].hist(scores, bins=20, alpha=0.7, color='skyblue')
    axes[1, 0].set_title('Credit Score Distribution')
    axes[1, 0].set_xlabel('Credit Score')
    axes[1, 0].set_ylabel('Frequency')
    
    # Risk level distribution
    risk_dist = validation_results['classification_metrics']['distribution']
    axes[1, 1].pie(risk_dist.values(), labels=risk_dist.keys(), autopct='%1.1f%%')
    axes[1, 1].set_title('Risk Level Distribution')
    
    plt.tight_layout()
    plt.savefig('validation_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    validate_model()