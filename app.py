from transformers import pipeline
import gradio as gr

# Load the text classification model
fraud_detector = pipeline("text-classification", model="bert-base-uncased")

def fraud_detection(transaction_details):
    prediction = fraud_detector(transaction_details)[0]
    label = prediction['label']
    confidence = prediction['score']
    #print(label, confidence)
    
    if label == 'LABEL_0':
        return f"Fraudulent. Confidence: {confidence:.2f}"
    else:
        return f"Legitimate. Confidence: {confidence:.2f}"



iface = gr.Interface(fn=fraud_detection, inputs="text", outputs="text", layout="vertical",
                     title="Fraud Detection System",
                     description="Enter transaction details to predict whether it's fraudulent or legitimate.",
                     examples=[
                         ["Fake investment opportunity promising high returns"],
                         ["Monthly rent payment to a landlord"],
                     ],
                     allow_flagging="never"
                     )

iface.launch()
