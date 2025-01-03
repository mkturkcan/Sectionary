import os
import logging
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import html2text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Chapter:
    """Represents a single chapter from an ebook."""
    number: int
    title: str
    content: str

    def get_filename(self) -> str:
        """Generate a sanitized filename for the chapter."""
        sanitized_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' 
                                 for c in self.title[:50]).strip()
        return f"chapter_{self.number:03d}_{sanitized_title}.txt"

class EPubConverter:
    """Handles the conversion of EPUB files to text."""
    
    def __init__(self):
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.html_converter.ignore_images = True

    def extract_text(self, epub_path: Path) -> Optional[str]:
        """Convert an EPUB file to plain text."""
        try:
            book = epub.read_epub(str(epub_path))
            chapters_content = []

            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    chapters_content.append(self.html_converter.handle(str(soup)))

            return '\n\n'.join(chapters_content)
        except Exception as e:
            logger.error(f"Failed to process {epub_path}: {str(e)}")
            return None

class ChapterSplitter:
    """Handles the splitting of text content into chapters."""
    
    def __init__(self, min_line_length: int = 30, min_chapter_lines: int = 3,
                 min_chapter_chars: int = 100):
        self.min_line_length = min_line_length
        self.min_chapter_lines = min_chapter_lines
        self.min_chapter_chars = min_chapter_chars

    def _extract_chapter_title(self, content: str) -> str:
        """Extract a title from the chapter content."""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return "untitled"
            
        first_line = lines[0]
        if first_line.startswith('#'):
            return first_line.strip('# ')
        return first_line[:50]  # Use first 50 chars of first line

    def _is_valid_chapter(self, content: str) -> bool:
        """Check if the content meets minimum chapter requirements."""
        return (len(content.split('\n')) > self.min_chapter_lines or
                len(content) > self.min_chapter_chars)

    def split_chapters(self, content: str) -> List[Chapter]:
        """Split text content into chapters."""
        if not content:
            return []

        lines = content.split('\n')
        chapters = []
        current_chapter = []
        empty_line_count = 0

        for line in lines:
            line = line.rstrip()

            # Handle chapter breaks
            if line.startswith('#') and current_chapter:
                chapter_text = '\n'.join(current_chapter).strip()
                if self._is_valid_chapter(chapter_text):
                    chapters.append(chapter_text)
                current_chapter = []
                empty_line_count = 0

            # Handle multiple empty lines as chapter breaks
            if not line:
                empty_line_count += 1
                if empty_line_count >= 3 and current_chapter:
                    chapter_text = '\n'.join(current_chapter).strip()
                    if self._is_valid_chapter(chapter_text):
                        chapters.append(chapter_text)
                    current_chapter = []
                    empty_line_count = 0
            else:
                empty_line_count = 0

            current_chapter.append(line)

        # Handle last chapter
        if current_chapter:
            chapter_text = '\n'.join(current_chapter).strip()
            if self._is_valid_chapter(chapter_text):
                chapters.append(chapter_text)

        # Create Chapter objects
        return [
            Chapter(
                number=i + 1,
                title=self._extract_chapter_title(content),
                content=content
            )
            for i, content in enumerate(chapters)
        ]

class EPubProcessor:
    """Main class for processing EPUB files and saving chapters."""
    
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.converter = EPubConverter()
        self.splitter = ChapterSplitter()

    def process_single_book(self, epub_path: Path) -> int:
        """Process a single EPUB file and return number of chapters processed."""
        logger.info(f"Processing: {epub_path}")
        
        # Create book-specific output directory
        book_dir = self.output_dir / epub_path.stem
        book_dir.mkdir(parents=True, exist_ok=True)

        # Convert and split into chapters
        content = self.converter.extract_text(epub_path)
        if not content:
            return 0

        chapters = self.splitter.split_chapters(content)
        
        # Save chapters
        for chapter in chapters:
            chapter_path = book_dir / chapter.get_filename()
            try:
                chapter_path.write_text(chapter.content, encoding='utf-8')
                logger.info(f"Saved: {chapter_path}")
            except Exception as e:
                logger.error(f"Failed to save chapter {chapter.number}: {str(e)}")

        return len(chapters)

    def process_all_books(self) -> tuple[int, int]:
        """Process all EPUB files in the input directory."""
        total_books = 0
        total_chapters = 0

        for epub_path in self.input_dir.rglob('*.epub'):
            num_chapters = self.process_single_book(epub_path)
            if num_chapters > 0:
                total_books += 1
                total_chapters += num_chapters

        return total_books, total_chapters
