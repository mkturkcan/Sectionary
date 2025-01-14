import openai
import pandas as pd
import time
import concurrent.futures
import pandas as pd
import openai
from time import sleep
from tqdm import tqdm

def setup_openai_client(api_key):
    """Initialize the OpenAI client with the provided API key."""
    openai.api_key = api_key
    return openai.Client(api_key=api_key)

# Set your OpenAI API key
api_key = ''

client = setup_openai_client(api_key)
def clean_text_with_openai(text):
    """
    Uses OpenAI's API (gpt-4o-mini or similar) to clean text according to:
    - Remove Markdown formatting and chapter titles
    - Combine lines that were incorrectly split
    - Remove any non-Latin characters that don't belong to the input
    - Fix punctuation, typos, corrupted words, etc.
    - Keep content the same otherwise
    """
    system_message = """You are a text cleaning assistant. Your task is to:
    1. Remove Markdown formatting and chapter titles
    2. Combine incorrectly split lines
    3. Remove non-Latin characters that don't belong
    4. Fix punctuation and typos
    5. Repair corrupted words
    
    Maintain the original meaning and content. Return only the cleaned text without any explanations."""

    
    user_prompt = f"""{text}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    # Extract and return the cleaned text
    cleaned_text = response.choices[0].message.content.strip()
    return cleaned_text

def main():
    # Read your CSV into a pandas DataFrame
    df = pd.read_csv("all_generations_merged.csv")  # Update with your CSV file
    
    # Specify the column you want to clean
    column_to_clean = "chosen"  # Update with the name of the column
    
    # Define a batch size to process rows in chunks
    batch_size = 10
    
    # Loop through the DataFrame in batches
    for start in range(0, len(df), batch_size):
        print(start)
        end = min(start + batch_size, len(df))
        
        # Collect batch indices
        batch_indices = df.index[start:end]
        
        # Extract the texts to be cleaned for this batch
        texts_to_clean = df.loc[batch_indices, column_to_clean].tolist()

        # Process the batch **concurrently** using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            cleaned_texts = list(executor.map(clean_text_with_openai, texts_to_clean))
        
        # Update the DataFrame with cleaned texts
        df.loc[batch_indices, column_to_clean] = cleaned_texts
        
        df.to_csv("cleaned_output.csv", index=False)
        
        # Sleep briefly to respect potential API rate limits
        time.sleep(1)

if __name__ == "__main__":
    main()
