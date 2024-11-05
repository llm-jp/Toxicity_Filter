import os
import onnxruntime as ort
from transformers import AutoTokenizer
import argparse
import json
import gzip
import time
import numpy as np
from tqdm import tqdm

def parse_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Batch inference using ONNX model with TensorRT.")

    parser.add_argument('--model_path', type=str, required=True, help="Path to the ONNX model.")
    parser.add_argument('--tokenizer_model', type=str, default="ku-nlp/deberta-v3-base-japanese", help="The tokenizer model to use.")
    parser.add_argument('--input_data_path', type=str, required=True, help="Path to the input gzipped JSONL file containing texts.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to the output CSV file.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for inference.")
    parser.add_argument('--max_length', type=int, default=256, help="Maximum sequence length for tokenization.")

    return parser.parse_args()

def load_onnx_model(model_path):
    """Load the ONNX model with TensorRT Execution Provider. Raise exception if not available."""
    available_providers = ort.get_available_providers()
    if 'TensorrtExecutionProvider' not in available_providers:
        raise RuntimeError("TensorrtExecutionProvider is unavailable. Ensure TensorRT is installed and properly configured.")
    
    session = ort.InferenceSession(model_path, providers=['TensorrtExecutionProvider'])
    return session

def tokenize_texts(tokenizer, texts, max_length=256):
    """Tokenize a list of texts."""
    return tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='np'
    )

def onnx_inference(session, input_ids, attention_mask):
    """Perform inference using ONNX runtime."""
    ort_inputs = {
        session.get_inputs()[0].name: input_ids,
        session.get_inputs()[1].name: attention_mask
    }
    ort_outs = session.run(None, ort_inputs)
    return ort_outs

def run_inference(session, tokenizer, texts, max_length):
    """Run inference on a list of texts and return logits."""
    tokenized = tokenize_texts(tokenizer, texts, max_length)
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']

    # Perform inference
    outputs = onnx_inference(session, input_ids, attention_mask)

    ## Assuming the model outputs logits for binary classification
    #logits = outputs[0].flatten()
    
    # Extract the positive class logit (index 1)
    # outputs[0] has shape (batch_size, 2)
    logits = outputs[0][:, 1]  # Shape: (batch_size,)

    return logits

def write_logits(outfile, logits):
    """Write the positive class logits to the output CSV file."""
    for logit in logits:
        outfile.write(f"{logit}\n")

def process_batches(session, tokenizer, input_path, output_path, batch_size, max_length):
    """Process the input data in batches and write the logits to the output CSV file."""
    # Determine if input is gzipped
    open_func = gzip.open if input_path.endswith('.gz') else open
    total_size = os.path.getsize(input_path)

    with open_func(input_path, 'rt', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile, \
         tqdm(total=total_size, unit='B', unit_scale=True, desc="Processing") as pbar:

        batch_texts = []
        total_processed = 0

        for line_number, line in enumerate(infile, start=1):
            pbar.update(len(line.encode('utf-8')))
            line = line.strip()
            if not line:
                # Empty line, write 'null'
                outfile.write("null\n")
                continue

            try:
                data = json.loads(line)
                text = data.get('text', '').strip()
                if text:
                    batch_texts.append(text)
                else:
                    # Missing or empty 'text' field
                    outfile.write("null\n")
            except json.JSONDecodeError:
                # Invalid JSON line
                outfile.write("null\n")
                continue

            # If batch is full, perform inference
            if len(batch_texts) == batch_size:
                logits = run_inference(session, tokenizer, batch_texts, max_length)
                write_logits(outfile, logits)
                total_processed += len(batch_texts)
                batch_texts = []

        # Process any remaining texts in the batch
        if batch_texts:
            logits = run_inference(session, tokenizer, batch_texts, max_length)
            write_logits(outfile, logits)
            total_processed += len(batch_texts)
            batch_texts = []

    print(f"\nTotal samples processed: {total_processed}")

def main():
    args = parse_arguments()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)

    # Load the ONNX model (raises exception if TensorRT provider is unavailable)
    session = load_onnx_model(args.model_path)

    # Start processing
    start_time = time.time()
    process_batches(
        session=session,
        tokenizer=tokenizer,
        input_path=args.input_data_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Total inference time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()

