"""
Experiment tracking and database management for ML experiments.

Stores results in SQLite database for later analysis and comparison.
"""

from typing import Dict, List, Optional, Any
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
import json


class ExperimentTracker:
    """
    Track ML experiments in SQLite database.
    
    Stores:
    - Experiment metadata
    - Model configurations
    - Performance metrics
    - Feature selections
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        experiment_name: str = 'experiment'
    ):
        if db_path is None:
            db_path = './experiments.db'
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.experiment_id = None
        
        # Initialize database
        self._init_db()
        
        # Create experiment record
        self._create_experiment()
    
    def _init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                created_at TEXT
            )
            """
        )
        
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                selector TEXT,
                estimator TEXT,
                feature_count INTEGER,
                metrics_json TEXT,
                metadata_json TEXT,
                created_at TEXT
            )
            """
        )
        
        conn.commit()
        conn.close()
    
    def _create_experiment(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO experiments (name, created_at) VALUES (?, ?)",
            (self.experiment_name, datetime.utcnow().isoformat())
        )
        conn.commit()
        self.experiment_id = cursor.lastrowid
        conn.close()
    
    def log_result(self, result: Dict[str, Any]):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metrics_json = json.dumps(result.get('metrics', {}))
        metadata_json = json.dumps(result.get('metadata', {}))
        
        cursor.execute(
            "INSERT INTO results (experiment_id, selector, estimator, feature_count, metrics_json, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                self.experiment_id,
                result.get('selector'),
                result.get('estimator'),
                result.get('feature_count'),
                metrics_json,
                metadata_json,
                datetime.utcnow().isoformat()
            )
        )
        conn.commit()
        conn.close()
    
    def get_results(
        self,
        experiment_id: Optional[int] = None
    ) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        query = "SELECT r.id, r.experiment_id, e.name as experiment_name, r.selector, r.estimator, r.feature_count, r.metrics_json, r.metadata_json, r.created_at FROM results r JOIN experiments e ON r.experiment_id = e.id"
        if experiment_id is not None:
            query += f" WHERE r.experiment_id = {int(experiment_id)}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return df
        
        # Expand json columns
        df['metrics'] = df['metrics_json'].apply(lambda x: json.loads(x) if x else {})
        df['metadata'] = df['metadata_json'].apply(lambda x: json.loads(x) if x else {})
        
        # Normalize metrics into columns for convenience
        metrics_df = pd.json_normalize(df['metrics'])
        df = pd.concat([df.drop(columns=['metrics_json','metadata_json','metrics','metadata']), metrics_df], axis=1)
        return df
    
    def list_experiments(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM experiments", conn)
        conn.close()
        return df
    
    def get_best_result(
        self,
        metric: str = 'test_accuracy_mean',
        experiment_id: Optional[int] = None
    ) -> Dict[str, Any]:
        df = self.get_results(experiment_id)
        if df.empty:
            return {}
        if metric not in df.columns:
            raise KeyError(f"Metric not found in results: {metric}")
        
        best_row = df.loc[df[metric].idxmax()]
        return best_row.to_dict()
    
    def export_to_csv(
        self,
        output_path: str,
        experiment_id: Optional[int] = None
    ):
        df = self.get_results(experiment_id)
        df.to_csv(output_path, index=False)



def init_experiment_db(db_path: str = './experiments.db'):
    """
    Initialize experiment database.
    
    Args:
        db_path: Path to database file
    """
    tracker = ExperimentTracker(db_path=db_path, experiment_name='init')
    print(f"Database initialized at: {db_path}")


def load_experiment_results(
    db_path: str = './experiments.db',
    experiment_id: Optional[int] = None
) -> pd.DataFrame:
    """
    Load results from database.
    
    Args:
        db_path: Path to database
        experiment_id: Specific experiment ID (None = latest)
        
    Returns:
        DataFrame with results
    """
    tracker = ExperimentTracker(db_path=db_path, experiment_name='temp')
    
    if experiment_id is None:
        # pick latest
        df_exps = tracker.list_experiments()
        if df_exps.empty:
            return pd.DataFrame()
        experiment_id = int(df_exps['id'].max())
    
    return tracker.get_results(experiment_id)


if __name__ == "__main__":
    # Example usage
    tracker = ExperimentTracker(
        db_path='./test_experiments.db',
        experiment_name='test_run'
    )
    
    # Log some dummy results
    for i in range(3):
        tracker.log_result({
            'selector': 'mutual_info',
            'estimator': 'random_forest',
            'feature_count': 50 + i,
            'metrics': {'test_accuracy_mean': 0.8 + i*0.01},
            'metadata': {'run_idx': i}
        })
    
    # Get results
    results = tracker.get_results()
    print("\nResults:")
    print(results[['selector', 'feature_count', 'estimator', 'test_accuracy_mean']])
    
    # Get best
    best = tracker.get_best_result('test_accuracy_mean')
    print(f"\nBest Result:")
    print(f"  Selector: {best['selector']}")
    print(f"  Features: {best['feature_count']}")
    print(f"  Estimator: {best['estimator']}")
    print(f"  Accuracy: {best['test_accuracy_mean']:.4f}")
