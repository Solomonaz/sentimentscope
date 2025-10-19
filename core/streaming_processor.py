import os
import json
import fitz
from typing import List, Dict, Any, Generator, Optional
from datetime import datetime

class StreamingProcessor:
    """
    Process large PDFs in a streaming fashion to minimize memory usage
    and enable progress tracking for very large documents
    """
    
    def __init__(self, llm_clients, analytics_engine, max_memory_mb: int = 100):
        self.llm_clients = llm_clients
        self.analytics_engine = analytics_engine
        self.max_memory_mb = max_memory_mb
    
    def process_large_pdf_streaming(self, pdf_path: str, output_path: str, 
                                  max_pages: int = None) -> Dict[str, Any]:
        """
        Process a PDF using streaming to handle very large documents
        
        Args:
            pdf_path: Path to PDF file
            output_path: Directory for output files
            max_pages: Maximum pages to process (None for all)
            
        Returns:
            Summary of processing results
        """
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        jsonl_path = os.path.join(output_path, f"{base_name}_analysis.jsonl")
        summary_path = os.path.join(output_path, f"{base_name}_summary.json")
        
        print(f"🔄 Starting streaming processing of {pdf_path}")
        
        total_paragraphs = 0
        total_pages = 0
        model_usage = {}
        all_emotions = []
        all_intents = []
        all_confidences = []
        
        # Open JSONL file for streaming output
        with open(jsonl_path, 'w', encoding='utf-8') as jsonl_file:
            # Process each page sequentially
            for page_result in self._stream_pages(pdf_path, max_pages):
                total_pages += 1
                
                for paragraph_result in page_result['paragraphs']:
                    total_paragraphs += 1
                    
                    # Write result immediately
                    jsonl_file.write(json.dumps(paragraph_result, ensure_ascii=False) + '\n')
                    jsonl_file.flush()  # Ensure data is written to disk
                    
                    # Collect statistics for summary
                    model_used = paragraph_result.get('model_used', 'unknown')
                    model_usage[model_used] = model_usage.get(model_used, 0) + 1
                    all_emotions.append(paragraph_result.get('emotion', 'Neutral'))
                    all_intents.append(paragraph_result.get('intent', 'Informative'))
                    all_confidences.append(paragraph_result.get('confidence', 0.5))
                
                print(f"   📄 Processed page {total_pages}, total paragraphs: {total_paragraphs}")
                
                # Optional: Force garbage collection every N pages
                if total_pages % 10 == 0:
                    import gc
                    gc.collect()
        
        # Generate summary from collected statistics
        summary = self._generate_summary(
            pdf_path, total_pages, total_paragraphs, model_usage,
            all_emotions, all_intents, all_confidences
        )
        
        # Save summary
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Streaming processing complete: {total_pages} pages, {total_paragraphs} paragraphs")
        return summary
    
    def _stream_pages(self, pdf_path: str, max_pages: int = None) -> Generator[Dict[str, Any], None, None]:
        """
        Generator that yields one page at a time for processing
        
        Yields:
            Dictionary with page data and analyzed paragraphs
        """
        doc = fitz.open(pdf_path)
        try:
            pages_to_process = min(len(doc), max_pages) if max_pages else len(doc)
            
            for page_num in range(pages_to_process):
                page = doc[page_num]
                
                # Extract text from this page only
                text = page.get_text()
                paragraphs = self._extract_paragraphs_from_text(text, page_num + 1)
                
                # Analyze paragraphs for this page
                analyzed_paragraphs = []
                for para_data in paragraphs:
                    analysis = self.llm_clients.analyze_text(para_data['text'])
                    analyzed_paragraph = {
                        **para_data,
                        **analysis
                    }
                    analyzed_paragraphs.append(analyzed_paragraph)
                
                yield {
                    'page_number': page_num + 1,
                    'paragraphs': analyzed_paragraphs
                }
                
                # Clean up to free memory
                del page
                del text
                del paragraphs
                del analyzed_paragraphs
                
        finally:
            doc.close()
    
    def _extract_paragraphs_from_text(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract paragraphs from page text"""
        import re
        
        # Split by multiple newlines (paragraph boundaries)
        raw_paragraphs = re.split(r'\n\s*\n', text)
        
        paragraphs = []
        for i, para_text in enumerate(raw_paragraphs):
            para_text = para_text.strip()
            if len(para_text) >= 10:  # Minimum length requirement
                paragraphs.append({
                    'page': page_num,
                    'block_id': len(paragraphs) + 1,
                    'text': para_text,
                    'text_snippet': para_text[:200] + "..." if len(para_text) > 200 else para_text
                })
        
        return paragraphs
    
    def _generate_summary(self, pdf_path: str, total_pages: int, total_paragraphs: int,
                        model_usage: Dict, emotions: List[str], intents: List[str],
                        confidences: List[float]) -> Dict[str, Any]:
        """Generate summary statistics from streaming processing"""
        from collections import Counter
        
        emotion_dist = dict(Counter(emotions))
        intent_dist = dict(Counter(intents))
        
        # Convert counts to percentages
        total = len(emotions)
        emotion_dist_pct = {k: v/total for k, v in emotion_dist.items()}
        intent_dist_pct = {k: v/total for k, v in intent_dist.items()}
        
        # Find dominant emotion
        dominant_emotion = max(emotion_dist.items(), key=lambda x: x[1])[0] if emotion_dist else "Neutral"
        
        return {
            "file_name": os.path.basename(pdf_path),
            "processed_at": datetime.now().isoformat(),
            "processing_mode": "streaming",
            "document_stats": {
                "total_pages": total_pages,
                "total_paragraphs": total_paragraphs,
                "dominant_emotion": dominant_emotion,
                "average_confidence": sum(confidences) / len(confidences) if confidences else 0.5,
                "model_usage": model_usage
            },
            "analytics": {
                "emotion_distribution": emotion_dist_pct,
                "intent_distribution": intent_dist_pct,
                "confidence_stats": {
                    "mean": sum(confidences) / len(confidences) if confidences else 0.5,
                    "min": min(confidences) if confidences else 0.5,
                    "max": max(confidences) if confidences else 0.5
                }
            },
            "output_files": {
                "detailed_results": f"{os.path.splitext(os.path.basename(pdf_path))[0]}_analysis.jsonl",
                "summary": f"{os.path.splitext(os.path.basename(pdf_path))[0]}_summary.json"
            }
        }
    
    def generate_report_from_streaming(self, summary_data: Dict[str, Any], output_path: str):
        """Generate a PDF report from streaming processing summary"""
        try:
            from .modern_report_builder import ModernReportBuilder
            
            # Convert streaming summary to format expected by report builder
            report_data = self._convert_streaming_to_report_format(summary_data)
            
            report_builder = ModernReportBuilder()
            pdf_path = os.path.join(output_path, f"{os.path.splitext(summary_data['file_name'])[0]}_report.pdf")
            report_builder.generate_report(report_data, pdf_path)
            
            print(f"📊 Generated PDF report from streaming results: {pdf_path}")
            
        except Exception as e:
            print(f"⚠️  Could not generate PDF report from streaming results: {e}")
    
    def _convert_streaming_to_report_format(self, summary_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert streaming summary to standard report format"""
        return {
            "file_name": summary_data["file_name"],
            "processed_at": summary_data["processed_at"],
            "document_stats": summary_data["document_stats"],
            "analytics": summary_data["analytics"],
            "analysis": [],  # Empty for streaming mode - detailed data in JSONL
            "process_log": [
                {
                    "timestamp": summary_data["processed_at"],
                    "event": "streaming_processing_complete",
                    "detail": f"Processed {summary_data['document_stats']['total_pages']} pages via streaming"
                }
            ]
        }