from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    files=sorted(Path("results/tables").glob("final_metrics_test_uid_*.csv"))
    df=pd.read_csv(files[-1])
    print(df[["model","roc_auc","pr_auc","brier","recall_sensitivity","specificity","false_positive_rate"]])
