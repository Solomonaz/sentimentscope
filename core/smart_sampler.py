import numpy as np
from typing import List, Dict, Any
from collections import Counter
import hashlib

class SmartSampler:
    """Intelligent sampling for large documents to maintain representativeness"""
    
    def __init__(self, max_samples: int = 200):
        self.max_samples = max_samples
    
    def sample_paragraphs(self, paragraphs: List[Dict], max_samples: int = None) -> List[Dict]:
        """
        Apply intelligent sampling to maintain document representativeness
        
        Args:
            paragraphs: List of paragraph dictionaries
            max_samples: Maximum number of samples to return
            
        Returns:
            Sampled list of paragraphs
        """
        if max_samples is None:
            max_samples = self.max_samples
            
        if len(paragraphs) <= max_samples:
            return paragraphs
        
        print(f"   📉 Applying smart sampling: {len(paragraphs)} -> {max_samples} paragraphs")
        
        # Strategy 1: Proportional sampling by page
        sampled_by_page = self._sample_by_page_proportional(paragraphs, max_samples)
        
        # Strategy 2: Ensure coverage of document structure
        sampled_with_coverage = self._ensure_document_coverage(sampled_by_page, paragraphs, max_samples)
        
        # Strategy 3: Prioritize content-rich paragraphs
        final_sampled = self._prioritize_content_rich(sampled_with_coverage, paragraphs, max_samples)
        
        # Strategy 4: Remove near-duplicates
        deduplicated = self._remove_near_duplicates(final_sampled, max_samples)
        
        print(f"   🔍 Sampling complete: {len(deduplicated)} unique paragraphs selected")
        return deduplicated
    
    def _sample_by_page_proportional(self, paragraphs: List[Dict], max_samples: int) -> List[Dict]:
        """Sample proportionally from each page"""
        # Group paragraphs by page
        pages = {}
        for para in paragraphs:
            page = para['page']
            if page not in pages:
                pages[page] = []
            pages[page].append(para)
        
        # Calculate samples per page based on proportion
        total_paragraphs = len(paragraphs)
        sampled = []
        
        for page_num, page_paragraphs in pages.items():
            proportion = len(page_paragraphs) / total_paragraphs
            samples_for_page = max(1, int(max_samples * proportion))
            
            # Use stratified sampling within page (beginning, middle, end)
            page_samples = self._sample_within_page(page_paragraphs, samples_for_page)
            sampled.extend(page_samples)
        
        return sampled
    
    def _sample_within_page(self, page_paragraphs: List[Dict], samples_needed: int) -> List[Dict]:
        """Sample from different sections of a single page"""
        if len(page_paragraphs) <= samples_needed:
            return page_paragraphs
        
        sampled = []
        page_len = len(page_paragraphs)
        
        # Take from beginning (30%)
        beginning_count = max(1, int(samples_needed * 0.3))
        sampled.extend(page_paragraphs[:beginning_count])
        
        # Take from middle (40%)
        middle_count = max(1, int(samples_needed * 0.4))
        middle_start = page_len // 2 - middle_count // 2
        middle_end = middle_start + middle_count
        sampled.extend(page_paragraphs[middle_start:middle_end])
        
        # Take from end (30%)
        end_count = samples_needed - len(sampled)
        if end_count > 0:
            sampled.extend(page_paragraphs[-end_count:])
        
        return sampled[:samples_needed]
    
    def _ensure_document_coverage(self, current_samples: List[Dict], all_paragraphs: List[Dict], 
                                max_samples: int) -> List[Dict]:
        """Ensure beginning, middle, and end of document are well represented"""
        sampled_set = set(self._get_paragraph_id(p) for p in current_samples)
        
        # Key document sections to ensure coverage
        sections = {
            'introduction': all_paragraphs[:10],  # First 10 paragraphs
            'conclusion': all_paragraphs[-10:],   # Last 10 paragraphs
            'middle': all_paragraphs[len(all_paragraphs)//2 - 5:len(all_paragraphs)//2 + 5]
        }
        
        enhanced_samples = current_samples.copy()
        
        for section_name, section_paragraphs in sections.items():
            for para in section_paragraphs:
                para_id = self._get_paragraph_id(para)
                if para_id not in sampled_set and len(enhanced_samples) < max_samples:
                    enhanced_samples.append(para)
                    sampled_set.add(para_id)
        
        return enhanced_samples[:max_samples]
    
    def _prioritize_content_rich(self, current_samples: List[Dict], all_paragraphs: List[Dict],
                               max_samples: int) -> List[Dict]:
        """Prioritize paragraphs with more substantial content"""
        if len(current_samples) >= max_samples:
            return current_samples[:max_samples]
        
        # Score paragraphs by content richness
        scored_paragraphs = []
        for para in all_paragraphs:
            if self._get_paragraph_id(para) in set(self._get_paragraph_id(p) for p in current_samples):
                continue
                
            score = self._calculate_content_score(para)
            scored_paragraphs.append((score, para))
        
        # Sort by score and take top ones
        scored_paragraphs.sort(reverse=True)
        additional_samples = [para for _, para in scored_paragraphs[:max_samples - len(current_samples)]]
        
        return current_samples + additional_samples
    
    def _calculate_content_score(self, paragraph: Dict) -> float:
        """Calculate content richness score for a paragraph"""
        text = paragraph.get('text', '')
        score = 0.0
        
        # Length factor (longer is generally better, but not too long)
        word_count = len(text.split())
        if 10 <= word_count <= 200:  # Ideal paragraph length
            score += min(word_count / 50, 1.0)
        
        # Diversity factor (more unique words)
        unique_words = len(set(text.lower().split()))
        score += unique_words / max(word_count, 1) * 0.5
        
        # Structural factors
        if any(marker in text for marker in [':', ';', '-']):  # Indicates complex structure
            score += 0.2
        
        return score
    
    def _remove_near_duplicates(self, paragraphs: List[Dict], max_samples: int) -> List[Dict]:
        """Remove near-duplicate paragraphs using text similarity"""
        if len(paragraphs) <= max_samples:
            return paragraphs
        
        unique_paragraphs = []
        seen_hashes = set()
        
        for para in paragraphs:
            # Create a hash based on the first 200 characters (captures essence)
            text_preview = para.get('text', '')[:200].lower()
            text_hash = hashlib.md5(text_preview.encode()).hexdigest()
            
            if text_hash not in seen_hashes:
                unique_paragraphs.append(para)
                seen_hashes.add(text_hash)
            
            if len(unique_paragraphs) >= max_samples:
                break
        
        return unique_paragraphs
    
    def _get_paragraph_id(self, paragraph: Dict) -> str:
        """Create a unique identifier for a paragraph"""
        return f"{paragraph['page']}_{paragraph['block_id']}_{hash(paragraph['text'][:100])}"
    
    def get_sampling_report(self, original_count: int, sampled_count: int) -> Dict[str, Any]:
        """Generate a report about the sampling process"""
        return {
            "original_paragraph_count": original_count,
            "sampled_paragraph_count": sampled_count,
            "sampling_ratio": sampled_count / original_count if original_count > 0 else 0,
            "reduction_percentage": (1 - sampled_count / original_count) * 100 if original_count > 0 else 0,
            "sampling_method": "smart_sampling_with_coverage"
        }