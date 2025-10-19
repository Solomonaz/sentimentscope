import os
import glob
import json
import fitz
from typing import List, Dict, Any, Optional
from datetime import datetime
from .extractor import PDFExtractor
from .llm_clients import LLMClients
from .analytics import AnalyticsEngine
from .modern_report_builder import ModernReportBuilder
import psutil
import time

class ResourceMonitor:
    """Monitor system resources during processing"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
    
    def memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except:
            return 0
    
    def processing_time_seconds(self) -> float:
        """Get processing time in seconds"""
        return time.time() - self.start_time
    
    def cpu_percent(self) -> float:
        """Get CPU usage percentage"""
        try:
            return self.process.cpu_percent()
        except:
            return 0

class ProductionConfig:
    """Production configuration with safety limits"""
    
    # Document limits
    MAX_PAGES = 50
    MAX_FILE_SIZE_MB = 10
    MAX_PARAGRAPHS = 200
    
    # Processing limits
    MAX_PROCESSING_TIME_MINUTES = 10
    BATCH_SIZE = 5
    
    # Memory limits
    MAX_MEMORY_MB = 512
    MEMORY_CHECK_INTERVAL = 10  # Check every 10 paragraphs
    
    # System limits
    MIN_FREE_DISK_SPACE_MB = 100
    
    @classmethod
    def validate_document(cls, file_path: str) -> Dict[str, Any]:
        """Validate document before processing"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'page_count': 0,
            'file_size_mb': 0
        }
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                validation_result['is_valid'] = False
                validation_result['errors'].append("File does not exist")
                return validation_result
            
            # Check file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            validation_result['file_size_mb'] = file_size_mb
            
            if file_size_mb > cls.MAX_FILE_SIZE_MB:
                validation_result['is_valid'] = False
                validation_result['errors'].append(
                    f"File too large: {file_size_mb:.1f}MB > {cls.MAX_FILE_SIZE_MB}MB limit"
                )
            
            # Check if it's a valid PDF
            try:
                doc = fitz.open(file_path)
                page_count = len(doc)
                validation_result['page_count'] = page_count
                doc.close()
                
                if page_count > cls.MAX_PAGES:
                    validation_result['is_valid'] = False
                    validation_result['errors'].append(
                        f"Document too long: {page_count} pages > {cls.MAX_PAGES} page limit"
                    )
                
                if page_count == 0:
                    validation_result['is_valid'] = False
                    validation_result['errors'].append("Document has no pages")
                    
            except Exception as e:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Invalid PDF file: {str(e)}")
            
            # Check disk space
            if not cls._has_sufficient_disk_space():
                validation_result['is_valid'] = False
                validation_result['errors'].append("Insufficient disk space")
        
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    @classmethod
    def _has_sufficient_disk_space(cls) -> bool:
        """Check if there's sufficient disk space"""
        try:
            disk_usage = psutil.disk_usage('.')
            free_space_mb = disk_usage.free / (1024 * 1024)
            return free_space_mb > cls.MIN_FREE_DISK_SPACE_MB
        except:
            return True  # If we can't check, assume it's ok

class PDFOrchestrator:
    """Orchestrates the entire PDF analysis workflow with production safeguards"""
    
    def __init__(self):
        self.extractor = PDFExtractor()
        self.llm_clients = LLMClients()
        self.analytics = AnalyticsEngine()
        self.report_builder = ModernReportBuilder()
        self.config = ProductionConfig()
        self.resource_monitor = ResourceMonitor()
        self.process_log = []
        
    def pre_validate(self, pdf_path: str) -> Dict[str, Any]:
        """Pre-validation before processing"""
        return self.config.validate_document(pdf_path)
    
    def process_pdf(self, pdf_path: str, output_path: str) -> Optional[Dict[str, Any]]:
        """Process a single PDF file with comprehensive error handling"""
        print(f"🔍 Processing: {os.path.basename(pdf_path)}")
        
        # Step 1: Pre-validation
        validation = self.pre_validate(pdf_path)
        if not validation['is_valid']:
            error_msg = f"Validation failed: {', '.join(validation['errors'])}"
            print(f"   ❌ {error_msg}")
            self._log_event('validation_failed', error_msg)
            raise ValueError(error_msg)
        
        print(f"   ✅ Validation passed: {validation['page_count']} pages, {validation['file_size_mb']:.1f}MB")
        
        try:
            # Step 2: Extract text with monitoring
            paragraphs = self._extract_text_safely(pdf_path)
            if not paragraphs:
                raise ValueError("No text could be extracted from PDF")
            
            print(f"   📄 Extracted {len(paragraphs)} paragraphs")
            
            # Step 3: Apply sampling if document is large
            if len(paragraphs) > self.config.MAX_PARAGRAPHS:
                paragraphs = self._apply_smart_sampling(paragraphs)
                print(f"   📊 Applied sampling: analyzing {len(paragraphs)} representative paragraphs")
            
            # Step 4: Analyze paragraphs with resource monitoring
            analysis_results, model_usage = self._analyze_paragraphs_safely(paragraphs)
            
            print(f"   🤖 Analysis complete. Models used: {model_usage}")
            
            # Step 5: Generate analytics
            analytics_data = self.analytics.compute_analytics(analysis_results)
            
            # Step 6: Prepare final output
            output_data = self._prepare_output_data(
                pdf_path, analysis_results, analytics_data, model_usage
            )
            
            # Step 7: Save outputs with error handling
            self._save_outputs_safely(output_data, pdf_path, output_path)
            
            print(f"   ✅ Processing completed successfully")
            return output_data
            
        except MemoryError as e:
            error_msg = f"Memory limit exceeded: {str(e)}"
            print(f"   💥 {error_msg}")
            self._log_event('memory_error', error_msg)
            raise
        except TimeoutError as e:
            error_msg = f"Processing timeout: {str(e)}"
            print(f"   ⏰ {error_msg}")
            self._log_event('timeout_error', error_msg)
            raise
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            print(f"   ❌ {error_msg}")
            self._log_event('processing_error', error_msg)
            raise
    
    def _extract_text_safely(self, pdf_path: str) -> List[Dict]:
        """Extract text with resource monitoring"""
        self._check_resources("before text extraction")
        
        try:
            paragraphs = self.extractor.extract_text(pdf_path)
            self._check_resources("after text extraction")
            return paragraphs
        except Exception as e:
            raise Exception(f"Text extraction failed: {str(e)}")
    
    def _apply_smart_sampling(self, paragraphs: List[Dict]) -> List[Dict]:
        """Apply intelligent sampling to large documents"""
        total_paragraphs = len(paragraphs)
        max_paragraphs = self.config.MAX_PARAGRAPHS
        
        if total_paragraphs <= max_paragraphs:
            return paragraphs
        
        print(f"   📉 Document large ({total_paragraphs} paragraphs), sampling to {max_paragraphs}")
        
        # Strategy: Take from beginning, middle, and end
        sampled = []
        
        # Take from beginning (20%)
        beginning_count = max(1, int(max_paragraphs * 0.2))
        sampled.extend(paragraphs[:beginning_count])
        
        # Take from middle (60%)
        middle_count = max(1, int(max_paragraphs * 0.6))
        middle_start = len(paragraphs) // 2 - middle_count // 2
        middle_end = middle_start + middle_count
        sampled.extend(paragraphs[middle_start:middle_end])
        
        # Take from end (20%)
        end_count = max_paragraphs - len(sampled)
        if end_count > 0:
            sampled.extend(paragraphs[-end_count:])
        
        # Remove duplicates and ensure we have exactly max_paragraphs
        unique_sampled = []
        seen_texts = set()
        
        for para in sampled:
            text_hash = hash(para['text'][:100])  # Hash first 100 chars to identify similar paragraphs
            if text_hash not in seen_texts and len(unique_sampled) < max_paragraphs:
                unique_sampled.append(para)
                seen_texts.add(text_hash)
        
        print(f"   🔍 Sampling complete: {len(unique_sampled)} unique paragraphs selected")
        return unique_sampled
    
    def _analyze_paragraphs_safely(self, paragraphs: List[Dict]) -> tuple:
        """Analyze paragraphs with comprehensive safety checks"""
        analysis_results = []
        model_usage = {}
        processed_count = 0
        
        for i, paragraph in enumerate(paragraphs):
            # Check resources every N paragraphs
            if processed_count % self.config.MEMORY_CHECK_INTERVAL == 0:
                self._check_resources(f"analyzing paragraph {processed_count + 1}")
            
            print(f"   Analyzing paragraph {processed_count + 1}/{len(paragraphs)}", end='\r')
            
            try:
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
                processed_count += 1
                
                # Log significant fallbacks
                if model_used == 'keyword_fallback':
                    self._log_event('keyword_fallback_used', f"Paragraph {paragraph['block_id']}")
            
            except Exception as e:
                # Log the error but continue with other paragraphs
                error_msg = f"Failed to analyze paragraph {paragraph['block_id']}: {str(e)}"
                print(f"   ⚠️  {error_msg}")
                self._log_event('paragraph_analysis_failed', error_msg)
                continue
        
        print()  # New line after progress indicator
        return analysis_results, model_usage
    
    def _check_resources(self, context: str = ""):
        """Check system resources and raise exceptions if limits exceeded"""
        memory_usage = self.resource_monitor.memory_usage_mb()
        processing_time = self.resource_monitor.processing_time_seconds()
        
        # Check memory
        if memory_usage > self.config.MAX_MEMORY_MB:
            raise MemoryError(
                f"Memory usage {memory_usage:.1f}MB exceeds limit {self.config.MAX_MEMORY_MB}MB "
                f"({context})"
            )
        
        # Check processing time
        max_time_seconds = self.config.MAX_PROCESSING_TIME_MINUTES * 60
        if processing_time > max_time_seconds:
            raise TimeoutError(
                f"Processing time {processing_time:.1f}s exceeds limit {max_time_seconds}s "
                f"({context})"
            )
        
        # Log resource usage periodically
        if int(processing_time) % 30 == 0:  # Every 30 seconds
            print(f"   📊 Resource check: {memory_usage:.1f}MB RAM, {processing_time:.1f}s elapsed")
    
    def _prepare_output_data(self, pdf_path: str, analysis_results: List[Dict], 
                           analytics_data: Dict, model_usage: Dict) -> Dict[str, Any]:
        """Prepare the final output data structure"""
        return {
            "file_name": os.path.basename(pdf_path),
            "processed_at": datetime.now().isoformat(),
            "document_stats": {
                "total_pages": max([p['page'] for p in analysis_results]) if analysis_results else 0,
                "total_sections": len(analysis_results),
                "dominant_emotion": analytics_data['dominant_emotion'],
                "average_confidence": analytics_data['average_confidence'],
                "model_usage": model_usage,
                "processing_time_seconds": self.resource_monitor.processing_time_seconds(),
                "memory_usage_mb": self.resource_monitor.memory_usage_mb()
            },
            "analysis": analysis_results,
            "analytics": analytics_data,
            "process_log": self.process_log,
            "validation_info": {
                "sampling_applied": len(analysis_results) < self.config.MAX_PARAGRAPHS,
                "resource_limits": {
                    "max_pages": self.config.MAX_PAGES,
                    "max_memory_mb": self.config.MAX_MEMORY_MB,
                    "max_processing_minutes": self.config.MAX_PROCESSING_TIME_MINUTES
                }
            }
        }
    
    def _save_outputs_safely(self, output_data: Dict[str, Any], pdf_path: str, output_path: str):
        """Save outputs with comprehensive error handling"""
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Save JSON output (critical - should always work if we got this far)
        json_filename = f"{output_path}/{base_name}_analysis.json"
        try:
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"   💾 JSON output saved: {json_filename}")
        except Exception as e:
            raise Exception(f"Failed to save JSON output: {str(e)}")
        
        # Generate PDF report (non-critical - can fail without stopping)
        pdf_report_filename = f"{output_path}/{base_name}_report.pdf"
        try:
            self._check_resources("before PDF generation")
            self.report_builder.generate_report(output_data, pdf_report_filename)
            print(f"   📊 PDF report generated: {pdf_report_filename}")
        except Exception as e:
            print(f"   ⚠️  PDF report generation failed: {e}")
            # Don't raise - JSON output is the primary product
    
    def _log_event(self, event: str, detail: str = ""):
        """Log processing events"""
        self.process_log.append({
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'detail': detail,
            'memory_mb': self.resource_monitor.memory_usage_mb(),
            'processing_time_seconds': self.resource_monitor.processing_time_seconds()
        })

def process_pdf_folder(input_path: str, output_path: str) -> List[Dict[str, Any]]:
    """Process all PDF files in the input folder with production-grade error handling"""
    orchestrator = PDFOrchestrator()
    
    # Find all PDF files
    pdf_pattern = os.path.join(input_path, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    
    if not pdf_files:
        print(f"📭 No PDF files found in {input_path}")
        return []
    
    print(f"🔍 Found {len(pdf_files)} PDF files to process")
    
    results = []
    successful = 0
    failed = 0
    
    for pdf_file in pdf_files:
        try:
            result = orchestrator.process_pdf(pdf_file, output_path)
            if result:
                results.append(result)
                successful += 1
                print(f"✅ Successfully processed: {os.path.basename(pdf_file)}")
                
        except ValueError as e:
            # Validation errors - expected, don't count as system failures
            failed += 1
            print(f"❌ Skipped {os.path.basename(pdf_file)}: {e}")
            continue
        except (MemoryError, TimeoutError) as e:
            # Resource limit errors - system protection working
            failed += 1
            print(f"🛑 Resource limit exceeded for {os.path.basename(pdf_file)}: {e}")
            continue
        except Exception as e:
            # Unexpected errors
            failed += 1
            print(f"💥 Unexpected error processing {os.path.basename(pdf_file)}: {e}")
            continue
    
    # Summary
    print(f"\n📊 Processing Summary:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed/Skipped: {failed}")
    print(f"   📄 Total Results: {len(results)}")
    
    return results