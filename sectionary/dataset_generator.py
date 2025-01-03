import os
from typing import List, Dict, Optional, Set
import pandas as pd
import tiktoken
from tqdm import tqdm
from transformers import pipeline
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# Constants
MAX_TOKENS_DEFAULT = 3000
MAX_SUMMARY_LENGTH = 150
MIN_SUMMARY_LENGTH = 30
MODEL_NAME_SUMMARY = "facebook/bart-large-cnn"
MODEL_NAME_GENERATION = "lemon07r/Gemma-2-Ataraxy-v4d-9B"

class TokenProcessor:
    """Handles text tokenization and chunking operations."""
    
    @staticmethod
    def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
        """Count tokens in text using tiktoken."""
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))

    @staticmethod
    def split_into_paragraphs(text: str) -> List[str]:
        """Split text into paragraphs based on double newlines."""
        return [p.strip() for p in text.split('\n\n') if p.strip()]

    @classmethod
    def chunk_chapter(cls, text: str, max_tokens: int = MAX_TOKENS_DEFAULT) -> List[str]:
        """Split chapter into chunks, respecting paragraph boundaries where possible."""
        paragraphs = cls.split_into_paragraphs(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = cls.count_tokens(paragraph)
            
            if paragraph_tokens > max_tokens:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                sentences = [s.strip() for s in paragraph.split('. ') if s.strip()]
                temp_chunk = []
                temp_tokens = 0
                
                for sentence in sentences:
                    sentence_tokens = cls.count_tokens(sentence)
                    if temp_tokens + sentence_tokens <= max_tokens:
                        temp_chunk.append(sentence)
                        temp_tokens += sentence_tokens
                    else:
                        if temp_chunk:
                            chunks.append('. '.join(temp_chunk) + '.')
                        temp_chunk = [sentence]
                        temp_tokens = sentence_tokens
                
                if temp_chunk:
                    chunks.append('. '.join(temp_chunk) + '.')
                
            elif current_tokens + paragraph_tokens <= max_tokens:
                current_chunk.append(paragraph)
                current_tokens += paragraph_tokens
            else:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_tokens = paragraph_tokens
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks

class TextGenerator:
    """Handles text generation using language models."""
    
    def __init__(self, model_name: str = MODEL_NAME_GENERATION):
        """Initialize the text generator with a specific model."""
        self.model, self.tokenizer = self._setup_model(model_name)
        
    def _setup_model(self, model_name: str):
        """Set up the language model and tokenizer."""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True
        )
        FastLanguageModel.for_inference(model)
        
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="gemma",
            mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"}
        )
        
        return model, tokenizer
    
    def generate(self, prompt: str, max_length: int = 2000) -> str:
        """Generate text based on a prompt."""
        messages = [{"from": "human", "value": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        
        outputs = self.model.generate(
            input_ids=inputs,
            max_length=max_length,
            use_cache=True
        )
        
        full_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return full_output[0].split('model\n')[-1][:-1]

class BookProcessor:
    """Handles book processing operations."""
    
    def __init__(self):
        """Initialize the book processor."""
        self.summarizer = pipeline("summarization", model=MODEL_NAME_SUMMARY, device='cuda:1')
        self.generator = TextGenerator()
    
    def process_book(self, book_folder: str, force_reprocess: bool = False) -> Optional[pd.DataFrame]:
        """Process all chapters in a book folder."""
        if self._is_book_processed(book_folder) and not force_reprocess:
            print(f"Book in {book_folder} has already been processed. Skipping...")
            return None
            
        self._setup_folders(book_folder)
        chapter_files = self._get_chapter_files(book_folder, force_reprocess)
        
        if not chapter_files:
            print("All chapters already processed. Skipping...")
            return None
        
        return self._process_chapters(book_folder, chapter_files)
    
    def _is_book_processed(self, book_folder: str) -> bool:
        """Check if a book has been fully processed."""
        complete_file = os.path.join(book_folder, 'complete_generation_results.csv')
        return os.path.exists(complete_file)
    
    def _setup_folders(self, book_folder: str) -> None:
        """Create necessary folders for processing."""
        for subfolder in ['summaries', 'generation']:
            os.makedirs(os.path.join(book_folder, subfolder), exist_ok=True)
    
    def _get_chapter_files(self, book_folder: str, force_reprocess: bool) -> List[str]:
        """Get list of chapter files to process."""
        chapter_files = sorted([
            f for f in os.listdir(book_folder)
            if f.startswith('chapter_') and f.endswith('.txt')
        ])
        
        if not force_reprocess:
            processed_chapters = self._get_processed_chapters(
                os.path.join(book_folder, 'generation')
            )
            return [f for f in chapter_files if f not in processed_chapters]
        
        return chapter_files
    
    @staticmethod
    def _get_processed_chapters(generation_folder: str) -> Set[str]:
        """Get set of processed chapter names."""
        if not os.path.exists(generation_folder):
            return set()
        
        return {
            f.replace('generation_', '').replace('.csv', '.txt')
            for f in os.listdir(generation_folder)
            if f.startswith('generation_') and f.endswith('.csv')
        }
    
    def _process_chapters(self, book_folder: str, chapter_files: List[str]) -> pd.DataFrame:
        """Process multiple chapters and combine results."""
        all_data = []
        complete_results_path = os.path.join(book_folder, 'complete_generation_results.csv')
        
        try:
            existing_df = pd.read_csv(complete_results_path)
            all_data.append(existing_df)
        except Exception as e:
            print(f"Error loading existing results: {str(e)}")
        
        for chapter_file in chapter_files:
            input_path = os.path.join(book_folder, chapter_file)
            output_path = os.path.join(
                book_folder,
                'generation',
                f"generation_{chapter_file.replace('.txt', '.csv')}"
            )
            
            print(f"\nProcessing {chapter_file}...")
            df = self._process_single_chapter(input_path, output_path)
            all_data.append(df)
        
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(complete_results_path, index=False)
        return final_df
    
    def _process_single_chapter(self, input_path: str, output_path: str) -> pd.DataFrame:
        """Process a single chapter file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            chapter_text = f.read()
        
        chunks = TokenProcessor.chunk_chapter(chapter_text)
        chunk_groups = self._combine_chunks(chunks)
        
        summaries = [
            self.summarizer(
                chunk,
                max_length=MAX_SUMMARY_LENGTH,
                min_length=MIN_SUMMARY_LENGTH,
                do_sample=False
            )[0]['summary_text']
            for chunk in tqdm(chunks)
        ]
        
        summary_groups = self._combine_chunks(summaries)
        
        data = []
        for summaries_text, original_text in zip(summary_groups, chunk_groups):
            prompt = f"Write a book chapter with the following summary:\n\n{summaries_text}"
            generated_text = self.generator.generate(prompt)
            
            data.append({
                'prompt': prompt,
                'chosen': original_text,
                'rejected': generated_text
            })
            
            # Save intermediate results
            pd.DataFrame(data).to_csv(f"{output_path}_progress.csv", index=False)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        return df
    
    @staticmethod
    def _combine_chunks(chunks: List[str], n: int = 4) -> List[str]:
        """Combine chunks into groups of n."""
        return ['\n\n'.join(chunks[i:i+n]) for i in range(0, len(chunks), n)]
