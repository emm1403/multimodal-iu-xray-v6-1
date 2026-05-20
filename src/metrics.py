import math
import numpy as np

def safe_div(a,b):
    return np.nan if b == 0 else a/b

def binary_metrics_from_confusion(tn,fp,fn,tp):
    sens=safe_div(tp,tp+fn); spec=safe_div(tn,tn+fp); fpr=safe_div(fp,fp+tn); fnr=safe_div(fn,fn+tp)
    ppv=safe_div(tp,tp+fp); npv=safe_div(tn,tn+fn); acc=safe_div(tp+tn,tp+tn+fp+fn); f1=safe_div(2*tp,2*tp+fp+fn)
    denom=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); mcc=safe_div(tp*tn-fp*fn,denom)
    return dict(accuracy=acc,precision_ppv=ppv,npv=npv,sensitivity=sens,specificity=spec,fpr=fpr,fnr=fnr,f1=f1,mcc=mcc)
