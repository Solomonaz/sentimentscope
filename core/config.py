
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
