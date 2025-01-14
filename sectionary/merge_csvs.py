import pandas as pd
import os
from pathlib import Path

def collect_and_merge_csvs(root_folder):
    """
    Find and merge all generation CSV files in the books directory structure
    
    Args:
        root_folder (str): Root folder containing all processed books
        
    Returns:
        pd.DataFrame: Merged dataframe with additional book info columns
    """
    all_data = []
    
    # Walk through directory structure
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Look for generation CSV files
        generation_files = [f for f in filenames if f.startswith('generation_') and f.endswith('.csv')]
        
        for file in generation_files:
            try:
                # Get book info from path
                book_folder = Path(dirpath).parent.name
                author = book_folder.split(' - ')[0]
                title = book_folder.split(' - ')[1] if ' - ' in book_folder else 'Unknown'
                
                # Get chapter info from filename
                chapter_num = int(file.split('_')[2])  # Assumes format "generation_001_..."
                
                # Read the CSV
                df = pd.read_csv(os.path.join(dirpath, file))
                
                # Add book and chapter info
                df['author'] = author
                df['title'] = title
                df['chapter'] = chapter_num
                df['source_file'] = file
                df['book_folder'] = book_folder
                
                all_data.append(df)
                print(f"Processed {book_folder} - Chapter {chapter_num}")
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
    
    if not all_data:
        raise ValueError("No valid CSV files found")
        
    # Merge all dataframes
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # Add some useful columns
    merged_df['prompt_length'] = merged_df['prompt'].str.len()
    merged_df['chosen_length'] = merged_df['chosen'].str.len()
    merged_df['rejected_length'] = merged_df['rejected'].str.len()
    
    # Save merged data
    output_path = os.path.join(root_folder, 'all_generations_merged.csv')
    merged_df.to_csv(output_path, index=False)
    print(f"\nSaved merged data to {output_path}")
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(merged_df)}")
    print(f"Unique books: {merged_df['title'].nunique()}")
    print(f"Unique authors: {merged_df['author'].nunique()}")
    print("\nSamples per author:")
    print(merged_df['author'].value_counts())
    
    return merged_df

def verify_data_quality(df):
    """
    Perform basic quality checks on the merged data
    
    Args:
        df (pd.DataFrame): Merged dataframe to check
    """
    print("\nData Quality Check:")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print("\nMissing values found:")
        print(missing[missing > 0])
    else:
        print("No missing values found")
    
    # Check for empty strings
    empty_strings = (df == '').sum()
    if empty_strings.any():
        print("\nEmpty strings found:")
        print(empty_strings[empty_strings > 0])
    
    # Check length distributions
    print("\nLength statistics:")
    for col in ['prompt_length', 'chosen_length', 'rejected_length']:
        print(f"\n{col}:")
        print(df[col].describe())
    
    # Flag potential issues
    issues = []
    
    # Very short texts (might indicate truncation)
    short_texts = df[df['chosen_length'] < 100]
    if len(short_texts) > 0:
        issues.append(f"Found {len(short_texts)} samples with very short chosen texts")
    
    # Very long texts (might indicate concatenation issues)
    long_texts = df[df['chosen_length'] > 50000]
    if len(long_texts) > 0:
        issues.append(f"Found {len(long_texts)} samples with very long chosen texts")
    
    if issues:
        print("\nPotential issues found:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("\nNo major issues found")

if __name__ == "__main__":
    # Path to your books directory
    books_root = "books_chapterized"
    
    try:
        # Collect and merge all CSVs
        merged_df = collect_and_merge_csvs(books_root)
        
        # Verify data quality
        verify_data_quality(merged_df)
        
    except Exception as e:
        print(f"Error: {str(e)}")