import os
from huggingface_hub import InferenceClient
import pandas as pd
from tqdm import tqdm

HUGGINGFACE_TOKEN = "<YOUR_HF_TOKEN>"  # Replace <YOUR_HF_TOKEN> with your HuggingFace token
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"  # Replace with desired model name
INPUT_FILE = "Models Results.xlsx"  # Input Excel file containing sentences
SHEET_NAME = "Mixtral - Setting1"  # Sheet name in the Excel file
OUTPUT_FILE = "moral_analysis_results.csv"  # Output CSV file
PROMPT_TEMPLATE = "Does the sentence - [Input_sentence] - convey a moral content? (Answer with one word: yes or no) If yes, what moral values does the text convey? (Categorize text only with one or more of these labels: Care/Harm, Fairness/Cheating, Loyalty/Betrayal, Authority/Subversion, Purity/Degradation)."

# Initialize Hugging Face Inference Client
client = InferenceClient(token=HUGGINGFACE_TOKEN)

if not os.path.exists(OUTPUT_FILE):
    pd.DataFrame(columns=["text", "label", "response", "corpus"]).to_csv(OUTPUT_FILE, index=False)

try:
    df = pd.read_excel(INPUT_FILE, SHEET_NAME)
except Exception as e:
    raise FileNotFoundError(f"Error reading input file: {e}")

for i, r in tqdm(df.iterrows(), total=len(df)):
    # Replace placeholder in the prompt with the actual sentence
    text = PROMPT_TEMPLATE.replace("[Input_sentence]", r['Text original'])

    messages = [
        {"role": "user", "content": text}
    ]

    try:
        completion = client.chat_completion(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=250,
            temperature=0.10,  
            top_p=0.90  
        )
        prediction = completion.choices[0].message.content
    except Exception as e:
        prediction = f"Error: {e}"

    new_row = {
        "text": r['Text original'],
        "label": r['label'],
        "response": prediction,
        "corpus": r['corpus']
    }

    pd.DataFrame([new_row]).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
