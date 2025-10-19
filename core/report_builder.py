from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
from typing import Dict, Any
import os

class SimpleReportBuilder:
    """Simple PDF report generator without charts"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
    
    def generate_report(self, analysis_data: Dict[str, Any], output_path: str):
        """Generate simple PDF report"""
        doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=0.5*inch)
        story = []
        
        # Title
        title = Paragraph("SentimentScope Analysis Report", self.styles['Heading1'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        # File info
        story.append(Paragraph(f"File: {analysis_data['file_name']}", self.styles['Heading2']))
        processed_date = datetime.fromisoformat(analysis_data['processed_at'].replace('Z', '+00:00'))
        story.append(Paragraph(f"Processed: {processed_date.strftime('%Y-%m-%d %H:%M')}", self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Key stats
        stats = analysis_data['document_stats']
        stats_data = [
            ["Metric", "Value"],
            ["Dominant Emotion", stats['dominant_emotion']],
            ["Average Confidence", f"{stats['average_confidence']:.2f}"],
            ["Total Pages", str(stats['total_pages'])],
            ["Total Sections", str(stats['total_sections'])],
            ["Analysis Model", list(stats.get('model_usage', {}).keys())[0] if stats.get('model_usage') else "Keyword Fallback"]
        ]
        
        stats_table = Table(stats_data, colWidths=[2*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495E')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ECF0F1')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(stats_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Emotion Distribution
        story.append(Paragraph("Emotion Distribution", self.styles['Heading2']))
        emotion_dist = analysis_data['analytics']['emotion_distribution']
        emotion_data = [["Emotion", "Percentage"]]
        for emotion, percentage in emotion_dist.items():
            emotion_data.append([emotion, f"{percentage:.1%}"])
        
        emotion_table = Table(emotion_data, colWidths=[1.5*inch, 1*inch])
        emotion_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(emotion_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Intent Distribution
        story.append(Paragraph("Intent Distribution", self.styles['Heading2']))
        intent_dist = analysis_data['analytics']['intent_distribution']
        intent_data = [["Intent", "Percentage"]]
        for intent, percentage in intent_dist.items():
            intent_data.append([intent, f"{percentage:.1%}"])
        
        intent_table = Table(intent_data, colWidths=[1.5*inch, 1*inch])
        intent_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(intent_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Sample analysis
        story.append(Paragraph("Sample Analysis", self.styles['Heading2']))
        sample_data = [["Page", "Emotion", "Intent", "Confidence", "Text Preview"]]
        
        for result in analysis_data['analysis'][:8]:  # First 8 results
            sample_data.append([
                str(result['page']),
                result['emotion'],
                result['intent'],
                f"{result.get('confidence', 0):.2f}",
                result.get('text_snippet', '')[:50] + "..."
            ])
        
        sample_table = Table(sample_data, colWidths=[0.5*inch, 1*inch, 1*inch, 1*inch, 2*inch])
        sample_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.whitesmoke])
        ]))
        
        story.append(sample_table)
        
        # Process log
        if analysis_data.get('process_log'):
            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph("Process Log", self.styles['Heading2']))
            for log in analysis_data['process_log'][-3:]:  # Last 3 entries
                log_text = f"{log.get('timestamp', '')}: {log.get('event', '')}"
                story.append(Paragraph(log_text, self.styles['Normal']))
                story.append(Spacer(1, 0.05*inch))
        
        doc.build(story)