from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime
from typing import Dict, Any, List
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

class ModernReportBuilder:
    """Modern PDF report generator with advanced analytics and insights"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_modern_styles()
        self.chart_temp_files = []
    
    def setup_modern_styles(self):
        """Setup modern paragraph styles"""
        # Modern color palette
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#2E8B57',
            'warning': '#FFA500',
            'danger': '#DC143C',
            'dark': '#2C3E50',
            'light': '#ECF0F1',
            'text': '#34495E'
        }
        
        # Custom styles
        custom_styles = {
            'ModernTitle': {
                'parent': self.styles['Heading1'],
                'fontSize': 24,
                'textColor': colors.HexColor(self.colors['primary']),
                'spaceAfter': 30,
                'alignment': 1
            },
            'ModernHeading1': {
                'parent': self.styles['Heading1'],
                'fontSize': 18,
                'textColor': colors.HexColor(self.colors['dark']),
                'spaceAfter': 20,
                'leftIndent': 10
            },
            'ModernHeading2': {
                'parent': self.styles['Heading2'],
                'fontSize': 14,
                'textColor': colors.HexColor(self.colors['secondary']),
                'spaceAfter': 12
            },
            'InsightBox': {
                'parent': self.styles['Normal'],
                'backColor': colors.HexColor(self.colors['light']),
                'borderColor': colors.HexColor(self.colors['primary']),
                'borderWidth': 1,
                'borderPadding': 10,
                'leftIndent': 10,
                'rightIndent': 10
            },
            'Recommendation': {
                'parent': self.styles['Normal'],
                'backColor': colors.HexColor('#E8F5E8'),
                'borderColor': colors.HexColor(self.colors['success']),
                'borderWidth': 1,
                'borderPadding': 8,
                'leftIndent': 8
            }
        }
        
        for style_name, style_config in custom_styles.items():
            if not hasattr(self.styles, style_name):
                self.styles.add(ParagraphStyle(name=style_name, **style_config))
    
    def generate_report(self, analysis_data: Dict[str, Any], output_path: str):
        """Generate modern PDF report with analytics and insights"""
        print(f"Generating modern PDF report: {output_path}")
        
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=0.5*inch)
            story = []
            
            # Cover page
            story.extend(self._create_modern_cover(analysis_data))
            story.append(Spacer(1, 0.5*inch))
            
            # Executive Summary with Insights
            story.extend(self._create_executive_summary(analysis_data))
            story.append(Spacer(1, 0.3*inch))
            
            # Advanced Analytics
            story.extend(self._create_advanced_analytics(analysis_data))
            story.append(Spacer(1, 0.3*inch))
            
            # Actionable Insights & Recommendations
            story.extend(self._create_actionable_insights(analysis_data))
            story.append(Spacer(1, 0.3*inch))
            
            # Detailed Analysis
            story.extend(self._create_detailed_analysis(analysis_data))
            
            doc.build(story)
            print(f"PDF report generated successfully: {output_path}")
            
        except Exception as e:
            print(f"Error generating PDF report: {e}")
            # Fallback to simple report
            self._generate_fallback_report(analysis_data, output_path)
        finally:
            # Clean up temporary chart files
            self._cleanup_temp_files()
    
    def _create_modern_cover(self, data: Dict[str, Any]) -> List[Any]:
        """Create modern cover page"""
        elements = []
        
        # Title with modern styling
        title = Paragraph("SENTIMENTSCOPE AI ANALYSIS", self.styles['ModernTitle'])
        elements.append(title)
        elements.append(Spacer(1, 0.3*inch))
        
        # File info in a modern layout
        processed_date = datetime.fromisoformat(data['processed_at'].replace('Z', '+00:00'))
        file_info = f"""
        <b>Document:</b> {data['file_name']}<br/>
        <b>Analysis Date:</b> {processed_date.strftime('%B %d, %Y at %H:%M')}<br/>
        <b>Total Pages:</b> {data['document_stats']['total_pages']} | <b>Sections Analyzed:</b> {data['document_stats']['total_sections']}
        """
        elements.append(Paragraph(file_info, self.styles['ModernHeading2']))
        elements.append(Spacer(1, 0.4*inch))
        
        # Dominant metrics in a styled table
        stats = data['document_stats']
        metrics_data = [
            ["KEY METRICS", "VALUE", "INSIGHT"],
            ["Dominant Emotion", stats['dominant_emotion'], self._get_emotion_insight(stats['dominant_emotion'])],
            ["Average Confidence", f"{stats['average_confidence']:.1%}", "High Reliability" if stats['average_confidence'] > 0.7 else "Moderate Reliability"],
            ["Analysis Model", list(stats.get('model_usage', {}).keys())[0] if stats.get('model_usage') else "AI-Powered", "Advanced AI Analysis"]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.colors['primary'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
        ]))
        
        elements.append(metrics_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Quick sentiment overview chart
        sentiment_chart = self._create_sentiment_gauge(data['document_stats']['average_confidence'])
        if sentiment_chart:
            elements.append(Paragraph("Analysis Confidence", self.styles['ModernHeading2']))
            elements.append(sentiment_chart)
        
        return elements
    
    def _create_executive_summary(self, data: Dict[str, Any]) -> List[Any]:
        """Create modern executive summary with insights"""
        elements = []
        
        elements.append(Paragraph("Executive Summary & Key Insights", self.styles['ModernHeading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Generate AI-powered insights
        insights = self._generate_ai_insights(data)
        
        # Overall assessment
        overall_text = f"""
        This comprehensive sentiment analysis reveals a document with <b>{insights['overall_tone']}</b> tone. 
        The analysis achieved <b>{data['document_stats']['average_confidence']:.1%} average confidence</b> across all sections, 
        indicating {'high reliability' if data['document_stats']['average_confidence'] > 0.7 else 'moderate reliability'} in the findings.
        """
        elements.append(Paragraph(overall_text, self.styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Key insights in boxes
        elements.append(Paragraph("Key Emotional Insights", self.styles['ModernHeading2']))
        for insight in insights['key_insights'][:3]:
            insight_text = f"• {insight}"
            elements.append(Paragraph(insight_text, self.styles['InsightBox']))
            elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Spacer(1, 0.2*inch))
        
        # Emotion distribution chart
        emotion_chart = self._create_emotion_distribution_chart(data['analytics']['emotion_distribution'])
        if emotion_chart:
            elements.append(Paragraph("Emotion Distribution", self.styles['ModernHeading2']))
            elements.append(emotion_chart)
            elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _create_advanced_analytics(self, data: Dict[str, Any]) -> List[Any]:
        """Create advanced analytics section with multiple charts"""
        elements = []
        
        elements.append(Paragraph("Advanced Analytics Dashboard", self.styles['ModernHeading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Emotion distribution
        emotion_chart = self._create_emotion_barchart(data['analytics']['emotion_distribution'])
        if emotion_chart:
            elements.append(Paragraph("Emotion Analysis", self.styles['ModernHeading2']))
            elements.append(emotion_chart)
            elements.append(Spacer(1, 0.2*inch))
        
        # Intent distribution
        intent_chart = self._create_intent_chart(data['analytics']['intent_distribution'])
        if intent_chart:
            elements.append(Paragraph("Intent Analysis", self.styles['ModernHeading2']))
            elements.append(intent_chart)
            elements.append(Spacer(1, 0.2*inch))
        
        # Confidence analysis
        confidence_data = [result.get('confidence', 0.5) for result in data['analysis']]
        confidence_chart = self._create_confidence_chart(confidence_data)
        if confidence_chart:
            elements.append(Paragraph("Confidence Distribution", self.styles['ModernHeading2']))
            elements.append(confidence_chart)
        
        return elements
    
    def _create_actionable_insights(self, data: Dict[str, Any]) -> List[Any]:
        """Create actionable insights and recommendations"""
        elements = []
        
        elements.append(Paragraph("Actionable Insights & Recommendations", self.styles['ModernHeading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        insights = self._generate_ai_insights(data)
        
        # Strategic recommendations
        elements.append(Paragraph("Strategic Recommendations", self.styles['ModernHeading2']))
        for i, recommendation in enumerate(insights['recommendations'][:5], 1):
            rec_text = f"<b>Recommendation {i}:</b> {recommendation}"
            elements.append(Paragraph(rec_text, self.styles['Recommendation']))
            elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Spacer(1, 0.2*inch))
        
        # Risk assessment
        elements.append(Paragraph("Risk Assessment", self.styles['ModernHeading2']))
        risk_factors = self._assess_risks(data)
        for risk in risk_factors:
            risk_text = f"• {risk}"
            elements.append(Paragraph(risk_text, self.styles['Normal']))
            elements.append(Spacer(1, 0.05*inch))
        
        return elements
    
    def _create_detailed_analysis(self, data: Dict[str, Any]) -> List[Any]:
        """Create detailed analysis section"""
        elements = []
        
        elements.append(Paragraph("Detailed Analysis", self.styles['ModernHeading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Create a modern table with analysis results
        detailed_data = [["Page", "Emotion", "Intent", "Confidence", "Key Phrases"]]
        
        for result in data['analysis'][:10]:  # Show first 10 for readability
            emotion_icon = self._get_emotion_icon(result['emotion'])
            detailed_data.append([
                str(result['page']),
                f"{emotion_icon} {result['emotion']}",
                result['intent'],
                f"{result.get('confidence', 0):.0%}",
                result.get('text_snippet', '')[:40] + "..."
            ])
        
        detailed_table = Table(detailed_data, colWidths=[0.5*inch, 1.2*inch, 1.2*inch, 1*inch, 2*inch])
        detailed_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.colors['dark'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
        ]))
        
        elements.append(detailed_table)
        
        if len(data['analysis']) > 10:
            elements.append(Spacer(1, 0.1*inch))
            elements.append(Paragraph(f"... and {len(data['analysis']) - 10} more sections analyzed", self.styles['Normal']))
        
        return elements
    
    def _create_sentiment_gauge(self, confidence: float) -> Any:
        """Create a simple sentiment gauge using table"""
        try:
            # Create a visual gauge using a table
            gauge_data = [["ANALYSIS CONFIDENCE"]]
            
            # Add gauge bars
            gauge_level = int(confidence * 10)
            gauge_bar = "█" * gauge_level + "░" * (10 - gauge_level)
            gauge_data.append([gauge_bar])
            gauge_data.append([f"{confidence:.1%}"])
            
            gauge_table = Table(gauge_data, colWidths=[4*inch])
            gauge_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.colors['primary'])),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 1), (-1, 1), 12),
                ('FONTSIZE', (0, 2), (-1, 2), 14),
                ('FONTNAME', (0, 2), (-1, 2), 'Helvetica-Bold'),
            ]))
            
            return gauge_table
            
        except Exception as e:
            print(f"Gauge creation failed: {e}")
            return None
    
    def _create_emotion_distribution_chart(self, emotion_dist: Dict[str, float]) -> Any:
        """Create emotion distribution as a table chart"""
        try:
            if not emotion_dist:
                return None
            
            # Create a styled table for emotion distribution
            emotion_data = [["EMOTION", "PERCENTAGE", "LEVEL"]]
            
            for emotion, percentage in emotion_dist.items():
                level = "High" if percentage > 0.3 else "Medium" if percentage > 0.1 else "Low"
                emotion_data.append([
                    emotion,
                    f"{percentage:.1%}",
                    level
                ])
            
            emotion_table = Table(emotion_data, colWidths=[1.5*inch, 1*inch, 1*inch])
            emotion_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.colors['secondary'])),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
            ]))
            
            return emotion_table
            
        except Exception as e:
            print(f"Emotion distribution table failed: {e}")
            return None
    
    def _create_emotion_barchart(self, emotion_dist: Dict[str, float]) -> Any:
        """Create a visual bar chart using table"""
        try:
            if not emotion_dist:
                return None
            
            # Create a visual bar chart using characters
            chart_data = [["EMOTION", "DISTRIBUTION"]]
            
            for emotion, percentage in emotion_dist.items():
                bar_length = int(percentage * 20)  # Scale to 20 characters max
                bar = "█" * bar_length
                chart_data.append([
                    emotion,
                    f"{bar} {percentage:.1%}"
                ])
            
            chart_table = Table(chart_data, colWidths=[1.5*inch, 3*inch])
            chart_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.colors['primary'])),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
            ]))
            
            return chart_table
            
        except Exception as e:
            print(f"Emotion barchart failed: {e}")
            return None
    
    def _create_intent_chart(self, intent_dist: Dict[str, float]) -> Any:
        """Create intent distribution chart"""
        try:
            if not intent_dist:
                return None
            
            intent_data = [["INTENT", "PERCENTAGE", "IMPACT"]]
            
            for intent, percentage in intent_dist.items():
                impact = "High" if intent == "Persuasive" else "Medium" if intent == "Informative" else "Low"
                intent_data.append([
                    intent,
                    f"{percentage:.1%}",
                    impact
                ])
            
            intent_table = Table(intent_data, colWidths=[1.5*inch, 1*inch, 1*inch])
            intent_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.colors['accent'])),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
            ]))
            
            return intent_table
            
        except Exception as e:
            print(f"Intent chart failed: {e}")
            return None
    
    def _create_confidence_chart(self, confidences: List[float]) -> Any:
        """Create confidence distribution chart"""
        try:
            if not confidences:
                return None
            
            # Calculate confidence statistics
            avg_confidence = np.mean(confidences)
            min_confidence = np.min(confidences)
            max_confidence = np.max(confidences)
            
            confidence_data = [
                ["STATISTIC", "VALUE"],
                ["Average Confidence", f"{avg_confidence:.1%}"],
                ["Minimum Confidence", f"{min_confidence:.1%}"],
                ["Maximum Confidence", f"{max_confidence:.1%}"],
                ["Reliability", "High" if avg_confidence > 0.7 else "Medium" if avg_confidence > 0.5 else "Low"]
            ]
            
            confidence_table = Table(confidence_data, colWidths=[2*inch, 2*inch])
            confidence_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.colors['success'])),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
            ]))
            
            return confidence_table
            
        except Exception as e:
            print(f"Confidence chart failed: {e}")
            return None
    
    def _generate_ai_insights(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered insights and recommendations"""
        dominant_emotion = data['document_stats']['dominant_emotion']
        emotion_dist = data['analytics']['emotion_distribution']
        intent_dist = data['analytics']['intent_distribution']
        avg_confidence = data['document_stats']['average_confidence']
        
        insights = {
            'overall_tone': self._assess_overall_tone(emotion_dist),
            'key_insights': [],
            'recommendations': []
        }
        
        # Generate key insights
        if emotion_dist.get('Optimistic', 0) > 0.4:
            insights['key_insights'].append("Document shows strong positive sentiment with optimistic outlook")
        if emotion_dist.get('Fearful', 0) > 0.3:
            insights['key_insights'].append("Significant cautious or fearful tones detected - consider risk mitigation")
        if emotion_dist.get('Angry', 0) > 0.2:
            insights['key_insights'].append("Presence of negative emotional tones that may require attention")
        
        if intent_dist.get('Persuasive', 0) > 0.5:
            insights['key_insights'].append("Document has strong persuasive intent - effective communication strategy")
        elif intent_dist.get('Informative', 0) > 0.6:
            insights['key_insights'].append("Primarily informative content with factual focus")
        
        # Generate recommendations based on analysis
        if dominant_emotion in ['Optimistic', 'Excited']:
            insights['recommendations'].extend([
                "Leverage positive tone for stakeholder communications",
                "Consider expanding on successful strategies mentioned", 
                "Use optimistic language in follow-up communications",
                "Capitalize on positive momentum for future initiatives"
            ])
        elif dominant_emotion in ['Fearful', 'Cautious']:
            insights['recommendations'].extend([
                "Address concerns and uncertainties proactively",
                "Develop risk mitigation strategies for mentioned challenges",
                "Provide reassurance in stakeholder communications",
                "Consider additional data to support decision-making"
            ])
        elif dominant_emotion in ['Angry', 'Sad']:
            insights['recommendations'].extend([
                "Address negative feedback or concerns immediately",
                "Consider tone adjustment for future communications", 
                "Implement feedback mechanisms for improvement",
                "Develop action plan to address underlying issues"
            ])
        
        # Confidence-based recommendations
        if avg_confidence < 0.6:
            insights['recommendations'].append("Consider manual review for low-confidence sections")
        
        # Intent-based recommendations
        if intent_dist.get('Persuasive', 0) > 0.4:
            insights['recommendations'].append("Strengthen persuasive elements with supporting data and evidence")
        
        # Ensure we have at least some insights
        if not insights['key_insights']:
            insights['key_insights'].append("Document shows balanced emotional tone across different sections")
        
        if not insights['recommendations']:
            insights['recommendations'].extend([
                "Maintain current communication strategy",
                "Continue monitoring emotional tone in future documents",
                "Consider periodic sentiment analysis for consistency"
            ])
        
        return insights
    
    def _assess_risks(self, data: Dict[str, Any]) -> List[str]:
        """Assess potential risks based on analysis"""
        risks = []
        emotion_dist = data['analytics']['emotion_distribution']
        avg_confidence = data['document_stats']['average_confidence']
        
        if emotion_dist.get('Fearful', 0) > 0.3:
            risks.append("High level of uncertainty or concern in content that may affect stakeholder confidence")
        if emotion_dist.get('Angry', 0) > 0.2:
            risks.append("Negative emotional tones detected that could impact document reception and effectiveness")
        if avg_confidence < 0.6:
            risks.append("Lower confidence scores may indicate ambiguous content requiring clarification")
        if emotion_dist.get('Neutral', 0) > 0.7:
            risks.append("Highly neutral tone may lack emotional engagement for target audience")
        
        if not risks:
            risks.append("No significant risks identified - document maintains appropriate tone and clarity")
        
        return risks
    
    def _assess_overall_tone(self, emotion_dist: Dict[str, float]) -> str:
        """Assess overall document tone"""
        positive_score = emotion_dist.get('Optimistic', 0) + emotion_dist.get('Excited', 0)
        negative_score = emotion_dist.get('Fearful', 0) + emotion_dist.get('Angry', 0) + emotion_dist.get('Sad', 0)
        
        if positive_score > negative_score + 0.2:
            return "predominantly positive and forward-looking"
        elif negative_score > positive_score + 0.2:
            return "predominantly cautious with concerns"
        elif emotion_dist.get('Neutral', 0) > 0.6:
            return "balanced and factual"
        else:
            return "mixed with varied emotional tones"
    
    def _get_emotion_insight(self, emotion: str) -> str:
        """Get insight for specific emotion"""
        insights = {
            'Optimistic': "Positive outlook, good for engagement and motivation",
            'Fearful': "Cautious tone, indicates areas needing attention or reassurance", 
            'Angry': "Strong negative sentiment, requires immediate addressing",
            'Sad': "Somber tone, may need supportive communication",
            'Neutral': "Balanced and factual, appropriate for formal communication",
            'Excited': "Enthusiastic tone, high energy and positive momentum",
            'Cautious': "Careful and measured approach, risk-aware positioning"
        }
        return insights.get(emotion, "Standard emotional tone for business communication")
    
    def _get_emotion_icon(self, emotion: str) -> str:
        """Get emotion icon (using text representation)"""
        icons = {
            'Optimistic': "↑", 'Fearful': "⚠", 'Angry': "⚡",
            'Sad': "↓", 'Neutral': "•", 'Excited': "★", 'Cautious': "ⓘ"
        }
        return icons.get(emotion, "•")
    
    def _generate_fallback_report(self, data: Dict[str, Any], output_path: str):
        """Generate a fallback simple report if modern one fails"""
        try:
            from core.simple_report_builder import SimpleReportBuilder
            fallback_builder = SimpleReportBuilder()
            fallback_builder.generate_report(data, output_path)
            print(f"Fallback PDF report generated: {output_path}")
        except Exception as e:
            print(f"Fallback report also failed: {e}")
    
    def _cleanup_temp_files(self):
        """Clean up temporary chart files"""
        for temp_file in self.chart_temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass  # Ignore cleanup errors