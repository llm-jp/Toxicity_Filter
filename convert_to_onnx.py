from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load your fine-tuned model and tokenizer
model_name = '../final_model'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Dummy input for tracing
dummy_input = tokenizer.encode_plus(
    "これはサンプルです。",
    return_tensors='pt',
    max_length=256,
    truncation=True,
    padding='max_length'
)

# Export the model to ONNX
torch.onnx.export(
    model,
    (dummy_input['input_ids'], dummy_input['attention_mask']),
    "deberta.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['output'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'output': {0: 'batch_size'}
    },
    opset_version=13
)

import onnx
import onnxruntime as ort

# Load the ONNX model
onnx_model = onnx.load("deberta.onnx")
onnx.checker.check_model(onnx_model)

# Inference with ONNX Runtime
ort_session = ort.InferenceSession(
    "deberta.onnx",
    providers=[
        'TensorrtExecutionProvider'
    ]
)

# Prepare inputs
inputs = {
    'input_ids': dummy_input['input_ids'].numpy(),
    'attention_mask': dummy_input['attention_mask'].numpy()
}

# Run inference
outputs = ort_session.run(None, inputs)
print(outputs)

