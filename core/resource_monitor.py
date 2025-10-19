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