class StreamingProcessor:
    """Process documents in streams to avoid memory issues"""
    
    def process_large_pdf(self, pdf_path: str, output_path: str):
        # Process page by page, writing results incrementally
        with open(f"{output_path}/partial_results.json", 'w') as f:
            f.write('{"analysis": [')
            
            first = True
            for page_data in self._stream_pages(pdf_path):
                if not first:
                    f.write(',')
                
                page_result = self._process_page(page_data)
                f.write(json.dumps(page_result))
                f.flush()  # Ensure data is written
                
                first = False
                
                # Memory cleanup
                del page_data
                del page_result
            
            f.write(']}')
    
    def _stream_pages(self, pdf_path: str):
        """Generator that yields one page at a time"""
        doc = fitz.open(pdf_path)
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                paragraphs = self._split_into_paragraphs(text)
                
                yield {
                    'page': page_num + 1,
                    'paragraphs': paragraphs
                }
        finally:
            doc.close()