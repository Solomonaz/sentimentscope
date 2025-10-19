#!/usr/bin/env python3
"""
SentimentScope - Emotion & Persuasive Intent PDF Analyzer
Abyss Widget Entry Point - Windows Compatible Version
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Main entry point for Abyss widget"""
    try:
        from core.orchestrator import process_pdf_folder
        
        # Get input path from environment variable
        pdf_input_path = os.environ.get('pdf_input_path', './input_pdfs/')
        output_path = './output/'
        
        print("SentimentScope Starting...")
        print(f"Input path: {pdf_input_path}")
        print(f"Output path: {output_path}")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Process all PDFs in the folder
        results = process_pdf_folder(pdf_input_path, output_path)
        
        print(f"Processing complete! Processed {len(results)} PDF files")
        
        return 0
        
    except Exception as e:
        print(f"Error in SentimentScope: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())