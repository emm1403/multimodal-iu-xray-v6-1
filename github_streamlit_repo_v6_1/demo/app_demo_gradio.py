from pathlib import Path
import pandas as pd
import gradio as gr
VERSION="v6_1"
PRIMARY_MODEL="image_calibrated"
PRIMARY_THRESHOLD_TYPE="min_fpr_recall_0.90"
BASE=Path(__file__).resolve().parent
err=pd.read_csv(BASE/f"error_analysis_test_{VERSION}.csv")
thr=pd.read_csv(BASE/f"thresholds_main_{VERSION}.csv")

def threshold_for(model):
    row=thr[(thr.model==model)&(thr.threshold_type==PRIMARY_THRESHOLD_TYPE)]
    return float(row.iloc[0].threshold)

def prob_col(model):
    return {"image_calibrated":"prob_image_cal","fusion_weighted":"prob_fusion_weighted","fusion_stack":"prob_fusion_stack","text_calibrated":"prob_text_cal"}[model]

def error_col(model):
    return "error_image_calibrated" if model=="image_calibrated" else f"error_{model}"

def demo(kind,model):
    e=error_col(model); df=err.copy()
    if kind=="true_positive": cand=df[(df.label==1)&(df[e]=="OK")]
    elif kind=="true_negative": cand=df[(df.label==0)&(df[e]=="OK")]
    elif kind=="false_positive": cand=df[df[e]=="FP"]
    elif kind=="false_negative": cand=df[df[e]=="FN"]
    else: cand=df
    if cand.empty: cand=df
    row=cand.sample(1,random_state=7).iloc[0]; t=threshold_for(model); p=float(row[prob_col(model)]); pred=int(p>=t)
    text=f"UID: {row.uid}\nArchivo: {row.filename}\nLabel real: {int(row.label)}\nModelo: {model}\nProbabilidad: {p:.3f}\nUmbral: {t:.3f}\nDecisión: {'ANORMALIDAD NO INCIDENTAL' if pred else 'NORMAL / NO RELEVANTE'}\nResultado: {row.get(e,'NA')}\n\nIndication: {row.get('indication','NA')}"
    img=row.get('image_path',None)
    return (img if isinstance(img,str) and Path(img).exists() else None), text

iface=gr.Interface(fn=demo,inputs=[gr.Dropdown(["true_positive","true_negative","false_positive","false_negative","random"],value="true_positive"),gr.Dropdown(["image_calibrated","fusion_weighted","fusion_stack","text_calibrated"],value=PRIMARY_MODEL)],outputs=[gr.Image(),gr.Textbox(lines=12)],title="Demo diagnóstico IU X-Ray V6.1")
if __name__=="__main__": iface.launch()
