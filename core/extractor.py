import fitz  # PyMuPDF
import PyPDF2
import os
from typing import List, Dict, Tuple
import re

class PDFExtractor:
    """Extract text from PDF files with page and paragraph information"""
    
    def __init__(self):
        self.min_paragraph_length = 10
    
    def extract_text(self, pdf_path: str) -> List[Dict]:
        """
        Extract text from PDF with page-wise paragraph segmentation
        
        Returns:
            List of dictionaries with page_num, paragraph_id, and text
        """
        try:
            return self._extract_with_pymupdf(pdf_path)
        except Exception as e:
            print(f"PyMuPDF failed, trying PyPDF2: {e}")
            return self._extract_with_pypdf2(pdf_path)
    
    def _extract_with_pymupdf(self, pdf_path: str) -> List[Dict]:
        """Extract using PyMuPDF (more accurate)"""
        doc = fitz.open(pdf_path)
        paragraphs = []
        paragraph_id = 1
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Split into paragraphs
            page_paragraphs = self._split_into_paragraphs(text)
            
            for para_text in page_paragraphs:
                if len(para_text.strip()) >= self.min_paragraph_length:
                    paragraphs.append({
                        'page': page_num + 1,
                        'block_id': paragraph_id,
                        'text': para_text.strip(),
                        'text_snippet': self._truncate_text(para_text.strip(), 200)
                    })
                    paragraph_id += 1
        
        doc.close()
        return paragraphs
    
    def _extract_with_pypdf2(self, pdf_path: str) -> List[Dict]:
        """Fallback extraction using PyPDF2"""
        paragraphs = []
        paragraph_id = 1
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                
                page_paragraphs = self._split_into_paragraphs(text)
                
                for para_text in page_paragraphs:
                    if len(para_text.strip()) >= self.min_paragraph_length:
                        paragraphs.append({
                            'page': page_num + 1,
                            'block_id': paragraph_id,
                            'text': para_text.strip(),
                            'text_snippet': self._truncate_text(para_text.strip(), 200)
                        })
                        paragraph_id += 1
        
        return paragraphs
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split by multiple newlines
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Further split by single newlines that likely indicate paragraphs
        refined_paragraphs = []
        for para in paragraphs:
            if '\n' in para and len(para) > 100:
                # This might be multiple paragraphs joined by single newlines
                sub_paras = re.split(r'(?<=[.!?])\s*\n', para)
                refined_paragraphs.extend(sub_paras)
            else:
                refined_paragraphs.append(para)
        
        return [p.strip() for p in refined_paragraphs if p.strip()]
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text for display purposes"""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."