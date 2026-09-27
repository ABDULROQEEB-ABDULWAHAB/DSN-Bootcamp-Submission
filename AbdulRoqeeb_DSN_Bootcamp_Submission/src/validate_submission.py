"""
Automated Submission Validation and Integrity Checker.
"""

import sys
import pandas as pd
import numpy as np

def validate_submission(submission_path='submissions/latest_submission.csv', sample_path='data/sample_submission.csv'):
    print(f"Validating submission: {submission_path}")
    
    sub = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_path)
    
    # 1. Check Row Count
    assert len(sub) == len(sample), f"Row count mismatch! Expected {len(sample)}, got {len(sub)}"
    print(f"  [PASS] Row count matches expected: {len(sub)}")
    
    # 2. Check Column Names
    assert list(sub.columns) == ['id', 'total_sales'], f"Columns mismatch! Expected ['id', 'total_sales'], got {list(sub.columns)}"
    print("  [PASS] Column headers match exact specification: ['id', 'total_sales']")
    
    # 3. Check ID alignment
    assert (sub['id'] == sample['id']).all(), "IDs do not match sample submission order or values!"
    print("  [PASS] ID order and values match sample_submission.csv exactly")
    
    # 4. Check for Nulls / NaNs / Infs
    null_count = sub['total_sales'].isnull().sum()
    assert null_count == 0, f"Found {null_count} null values in total_sales!"
    inf_count = np.isinf(sub['total_sales']).sum()
    assert inf_count == 0, f"Found {inf_count} infinite values in total_sales!"
    print("  [PASS] Zero nulls, NaNs, or infinite values detected")
    
    # 5. Check Non-negativity
    min_val = sub['total_sales'].min()
    assert min_val >= 0, f"Found negative sales predictions! Minimum value: {min_val}"
    print(f"  [PASS] Non-negativity constraint satisfied (min predicted sales: {min_val:.2f})")
    
    # 6. Check Distribution Realism
    mean_val = sub['total_sales'].mean()
    max_val = sub['total_sales'].max()
    print(f"  [INFO] Distribution stats: Mean={mean_val:.2f}, Max={max_val:.2f}, Std={sub['total_sales'].std():.2f}")
    assert 1800 <= mean_val <= 2500, f"Mean predicted value ({mean_val}) is outside expected retail bounds [1800, 2500]!"
    print("  [PASS] Distribution matches expected physical retail sales dynamics")
    
    print("\nALL SUBMISSION INTEGRITY CHECKS PASSED SUCCESSFULLY!")
    return True

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'submissions/latest_submission.csv'
    validate_submission(path)
