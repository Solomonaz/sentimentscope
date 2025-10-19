import os
import glob
import json
from typing import List, Dict, Any
from datetime import datetime
from .extractor import PDFExtractor
from .llm_clients import LLMClients
from .analytics import AnalyticsEngine
from .simple_report_builder import SimpleReportBuilder

class PDFOrchestrator:
    """Orchestrates the entire PDF analysis workflow"""
    
    def __init__(self):
        self.extractor = PDFExtractor()
        self.llm_clients = LLMClients()
        self.analytics = AnalyticsEngine()
        self.report_builder = SimpleReportBuilder()
        self.process_log = []
    
    def process_pdf(self, pdf_path: str, output_path: str) -> Dict[str, Any]:
        """Process a single PDF file"""
        print(f"Processing: {os.path.basename(pdf_path)}")
        
        # Extract text
        paragraphs = self.extractor.extract_text(pdf_path)
        print(f"   Extracted {len(paragraphs)} paragraphs")
        
        # Analyze each paragraph
        analysis_results = []
        model_usage = {}
        
        for i, paragraph in enumerate(paragraphs):
            print(f"   Analyzing paragraph {i+1}/{len(paragraphs)}", end='\r')
            
            # Analyze with LLM failover
            analysis = self.llm_clients.analyze_text(paragraph['text'])
            
            # Track model usage
            model_used = analysis.get('model_used', 'unknown')
            model_usage[model_used] = model_usage.get(model_used, 0) + 1
            
            # Combine with paragraph info
            result = {
                **paragraph,
                **analysis
            }
            analysis_results.append(result)
            
            # Log significant fallbacks
            if model_used == 'keyword_fallback':
                self.process_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'event': 'keyword_fallback_used',
                    'detail': f"Paragraph {paragraph['block_id']}"
                })
        
        print(f"\n   Analysis complete. Models used: {model_usage}")
        
        # Generate analytics
        analytics_data = self.analytics.compute_analytics(analysis_results)
        
        # Prepare final output
        output_data = {
            "file_name": os.path.basename(pdf_path),
            "processed_at": datetime.now().isoformat(),
            "document_stats": {
                "total_pages": max([p['page'] for p in analysis_results]) if analysis_results else 0,
                "total_sections": len(analysis_results),
                "dominant_emotion": analytics_data['dominant_emotion'],
                "average_confidence": analytics_data['average_confidence'],
                "model_usage": model_usage
            },
            "analysis": analysis_results,
            "analytics": analytics_data,
            "process_log": self.process_log
        }
        
        # Save JSON output first (this should always work)
        json_filename = f"{output_path}/{os.path.splitext(os.path.basename(pdf_path))[0]}_analysis.json"
        try:
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"   JSON output saved: {json_filename}")
        except Exception as e:
            print(f"   Failed to save JSON: {e}")
            return None
        
        # Generate PDF report (this might fail but shouldn't stop processing)
        pdf_report_filename = f"{output_path}/{os.path.splitext(os.path.basename(pdf_path))[0]}_report.pdf"
        try:
            self.report_builder.generate_report(output_data, pdf_report_filename)
            print(f"   PDF report generated: {pdf_report_filename}")
        except Exception as e:
            print(f"   Failed to generate PDF report: {e}")
            # Continue even if PDF fails
        
        return output_data

def process_pdf_folder(input_path: str, output_path: str) -> List[Dict[str, Any]]:
    """Process all PDF files in the input folder"""
    orchestrator = PDFOrchestrator()
    
    # Find all PDF files
    pdf_pattern = os.path.join(input_path, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    
    if not pdf_files:
        print(f"No PDF files found in {input_path}")
        return []
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    results = []
    for pdf_file in pdf_files:
        try:
            result = orchestrator.process_pdf(pdf_file, output_path)
            if result:  # Only add if processing was successful
                results.append(result)
        except Exception as e:
            print(f"Failed to process {pdf_file}: {e}")
            continue
    
    return results