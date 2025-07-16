import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeFiCreditScorer:
    """
    DeFi Credit Scoring Model for Aave V2 Protocol
    
    This model analyzes transaction patterns to assign credit scores (0-1000)
    based on reliability, risk behavior, and protocol usage patterns.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.score_bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess transaction data from JSON file."""
        logger.info(f"Loading data from {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        logger.info(f"Loaded {len(df)} transactions for {df['wallet'].nunique()} unique wallets")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from transaction data for each wallet."""
        logger.info("Engineering features...")
        
        features = []
        
        for wallet in df['wallet'].unique():
            wallet_data = df[df['wallet'] == wallet].copy()
            wallet_data = wallet_data.sort_values('timestamp')
            
            # Basic transaction features
            total_transactions = len(wallet_data)
            unique_actions = wallet_data['action'].nunique()
            
            # Time-based features
            first_tx = wallet_data['timestamp'].min()
            last_tx = wallet_data['timestamp'].max()
            account_age_days = (last_tx - first_tx).days + 1
            
            # Transaction frequency
            tx_frequency = total_transactions / max(account_age_days, 1)
            
            # Action distribution
            action_counts = wallet_data['action'].value_counts()
            deposit_count = action_counts.get('deposit', 0)
            borrow_count = action_counts.get('borrow', 0)
            repay_count = action_counts.get('repay', 0)
            redeem_count = action_counts.get('redeemunderlying', 0)
            liquidation_count = action_counts.get('liquidationcall', 0)
            
            # Repayment behavior
            repayment_ratio = repay_count / max(borrow_count, 1)
            
            # Liquidation risk
            liquidation_ratio = liquidation_count / max(total_transactions, 1)
            
            # Amount-based features
            total_volume = wallet_data['amount'].sum()
            avg_transaction_size = wallet_data['amount'].mean()
            max_transaction_size = wallet_data['amount'].max()
            
            # Deposit vs Borrow ratio
            deposit_amounts = wallet_data[wallet_data['action'] == 'deposit']['amount'].sum()
            borrow_amounts = wallet_data[wallet_data['action'] == 'borrow']['amount'].sum()
            deposit_borrow_ratio = deposit_amounts / max(borrow_amounts, 1)
            
            # Consistency metrics
            deposit_consistency = self._calculate_consistency(wallet_data, 'deposit')
            repay_consistency = self._calculate_consistency(wallet_data, 'repay')
            
            # Risk indicators
            large_tx_ratio = len(wallet_data[wallet_data['amount'] > wallet_data['amount'].quantile(0.9)]) / total_transactions
            
            # Time patterns
            hour_spread = wallet_data['timestamp'].dt.hour.nunique()
            day_spread = wallet_data['timestamp'].dt.day.nunique()
            
            # Recent activity
            recent_activity = len(wallet_data[wallet_data['timestamp'] > (last_tx - timedelta(days=30))])
            recent_activity_ratio = recent_activity / total_transactions
            
            # Health factor proxy (simplified)
            health_factor_proxy = (deposit_amounts + repay_count * avg_transaction_size) / max(borrow_amounts + liquidation_count * avg_transaction_size, 1)
            
            features.append({
                'wallet': wallet,
                'total_transactions': total_transactions,
                'unique_actions': unique_actions,
                'account_age_days': account_age_days,
                'tx_frequency': tx_frequency,
                'deposit_count': deposit_count,
                'borrow_count': borrow_count,
                'repay_count': repay_count,
                'redeem_count': redeem_count,
                'liquidation_count': liquidation_count,
                'repayment_ratio': repayment_ratio,
                'liquidation_ratio': liquidation_ratio,
                'total_volume': total_volume,
                'avg_transaction_size': avg_transaction_size,
                'max_transaction_size': max_transaction_size,
                'deposit_borrow_ratio': deposit_borrow_ratio,
                'deposit_consistency': deposit_consistency,
                'repay_consistency': repay_consistency,
                'large_tx_ratio': large_tx_ratio,
                'hour_spread': hour_spread,
                'day_spread': day_spread,
                'recent_activity_ratio': recent_activity_ratio,
                'health_factor_proxy': health_factor_proxy
            })
        
        feature_df = pd.DataFrame(features)
        logger.info(f"Engineered {len(feature_df.columns)-1} features for {len(feature_df)} wallets")
        return feature_df
    
    def _calculate_consistency(self, wallet_data: pd.DataFrame, action: str) -> float:
        """Calculate consistency of a specific action over time."""
        action_data = wallet_data[wallet_data['action'] == action]
        if len(action_data) < 2:
            return 0.0
        
        # Calculate time intervals between actions
        time_diffs = action_data['timestamp'].diff().dt.total_seconds().dropna()
        if len(time_diffs) == 0:
            return 0.0
        
        # Consistency is inverse of coefficient of variation
        cv = time_diffs.std() / max(time_diffs.mean(), 1)
        return 1 / (1 + cv)
    
    def create_target_scores(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Create target credit scores based on feature analysis."""
        logger.info("Creating target credit scores...")
        
        # Normalize features for scoring
        numeric_features = feature_df.select_dtypes(include=[np.number]).columns
        feature_normalized = feature_df.copy()
        
        for col in numeric_features:
            if col != 'wallet':
                feature_normalized[col] = (feature_df[col] - feature_df[col].min()) / (feature_df[col].max() - feature_df[col].min() + 1e-8)
        
        # Calculate composite score
        scores = []
        for idx, row in feature_normalized.iterrows():
            # Positive factors
            positive_score = (
                row['repayment_ratio'] * 200 +
                row['deposit_consistency'] * 150 +
                row['repay_consistency'] * 150 +
                row['health_factor_proxy'] * 100 +
                row['deposit_borrow_ratio'] * 100 +
                row['tx_frequency'] * 50 +
                row['account_age_days'] * 50 +
                row['recent_activity_ratio'] * 50
            )
            
            # Negative factors
            negative_score = (
                row['liquidation_count'] * 200 +
                row['liquidation_ratio'] * 300 +
                row['large_tx_ratio'] * 100
            )
            
            # Base score calculation
            base_score = positive_score - negative_score
            
            # Normalize to 0-1000 range
            final_score = max(0, min(1000, base_score))
            scores.append(final_score)
        
        feature_df['credit_score'] = scores
        return feature_df
    
    def train_model(self, feature_df: pd.DataFrame) -> None:
        """Train the credit scoring model."""
        logger.info("Training credit scoring model...")
        
        # Prepare features and target
        feature_cols = [col for col in feature_df.columns if col not in ['wallet', 'credit_score']]
        X = feature_df[feature_cols]
        y = feature_df['credit_score']
        
        # Handle missing values
        X = X.fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train ensemble model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        rf_model.fit(X_train_scaled, y_train)
        gb_model.fit(X_train_scaled, y_train)
        
        # Create ensemble
        rf_pred = rf_model.predict(X_test_scaled)
        gb_pred = gb_model.predict(X_test_scaled)
        ensemble_pred = (rf_pred + gb_pred) / 2
        
        # Store the best performing model
        self.model = {'rf': rf_model, 'gb': gb_model}
        
        # Calculate performance metrics
        mse = mean_squared_error(y_test, ensemble_pred)
        r2 = r2_score(y_test, ensemble_pred)
        
        logger.info(f"Model trained - MSE: {mse:.2f}, R²: {r2:.3f}")
        
        # Feature importance
        self.feature_importance = dict(zip(feature_cols, rf_model.feature_importances_))
        
    def predict_scores(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Predict credit scores for new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        feature_cols = [col for col in feature_df.columns if col != 'wallet']
        X = feature_df[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Ensemble prediction
        rf_pred = self.model['rf'].predict(X_scaled)
        gb_pred = self.model['gb'].predict(X_scaled)
        scores = (rf_pred + gb_pred) / 2
        
        # Ensure scores are within bounds
        scores = np.clip(scores, 0, 1000)
        
        result_df = feature_df[['wallet']].copy()
        result_df['credit_score'] = scores
        result_df['risk_level'] = result_df['credit_score'].apply(self._get_risk_level)
        
        return result_df
    
    def _get_risk_level(self, score: float) -> str:
        """Convert credit score to risk level."""
        if score >= 700:
            return 'Low'
        elif score >= 400:
            return 'Medium'
        else:
            return 'High'
    
    def analyze_results(self, results_df: pd.DataFrame, feature_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze scoring results and create distribution analysis."""
        logger.info("Analyzing results...")
        
        # Score distribution
        score_dist = pd.cut(results_df['credit_score'], bins=self.score_bins, include_lowest=True)
        distribution = score_dist.value_counts().sort_index()
        
        # Risk level distribution
        risk_dist = results_df['risk_level'].value_counts()
        
        # Feature analysis by risk level
        merged_df = results_df.merge(feature_df, on='wallet')
        
        analysis = {
            'score_distribution': distribution.to_dict(),
            'risk_distribution': risk_dist.to_dict(),
            'feature_importance': self.feature_importance,
            'high_risk_characteristics': self._analyze_risk_group(merged_df, 'High'),
            'low_risk_characteristics': self._analyze_risk_group(merged_df, 'Low'),
            'total_wallets': len(results_df)
        }
        
        return analysis
    
    def _analyze_risk_group(self, df: pd.DataFrame, risk_level: str) -> Dict[str, float]:
        """Analyze characteristics of a specific risk group."""
        group_df = df[df['risk_level'] == risk_level]
        
        if len(group_df) == 0:
            return {}
        
        characteristics = {
            'avg_liquidation_count': group_df['liquidation_count'].mean(),
            'avg_repayment_ratio': group_df['repayment_ratio'].mean(),
            'avg_tx_frequency': group_df['tx_frequency'].mean(),
            'avg_account_age': group_df['account_age_days'].mean(),
            'avg_deposit_consistency': group_df['deposit_consistency'].mean()
        }
        
        return characteristics

def main():
    """Main execution function."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python credit_scoring_model.py <path_to_transactions.json>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Initialize scorer
    scorer = DeFiCreditScorer()
    
    try:
        # Load and process data
        df = scorer.load_data(file_path)
        feature_df = scorer.engineer_features(df)
        
        # Create target scores for training
        feature_df_with_targets = scorer.create_target_scores(feature_df)
        
        # Train model
        scorer.train_model(feature_df_with_targets)
        
        # Predict scores
        results_df = scorer.predict_scores(feature_df)
        
        # Analyze results
        analysis = scorer.analyze_results(results_df, feature_df)
        
        # Save results
        results_df.to_csv('wallet_credit_scores.csv', index=False)
        
        # Save analysis
        with open('score_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info("Credit scoring completed successfully!")
        logger.info(f"Results saved to: wallet_credit_scores.csv")
        logger.info(f"Analysis saved to: score_analysis.json")
        
        # Print summary
        print("\n" + "="*50)
        print("DEFI CREDIT SCORING SUMMARY")
        print("="*50)
        print(f"Total wallets processed: {len(results_df)}")
        print(f"Average credit score: {results_df['credit_score'].mean():.1f}")
        print(f"Score distribution:")
        for risk, count in analysis['risk_distribution'].items():
            print(f"  {risk} risk: {count} wallets")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error in credit scoring: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()