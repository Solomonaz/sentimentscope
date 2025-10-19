from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
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
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

class ModernReportBuilder:
    """Modern PDF report generator with real plots, graphs and charts"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_modern_styles()
        self.chart_temp_files = []
        
        # Set modern plotting style
        plt.style.use('seaborn-v0_8')
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#2E8B57', 
                      '#FF6B6B', '#6A0572', '#FFD166', '#118AB2']
    
    def setup_modern_styles(self):
        """Setup modern paragraph styles"""
        # Modern color palette
        self.report_colors = {
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
                'textColor': colors.HexColor('#2E86AB'),
                'spaceAfter': 30,
                'alignment': 1
            },
            'ModernHeading1': {
                'parent': self.styles['Heading1'],
                'fontSize': 18,
                'textColor': colors.HexColor('#2C3E50'),
                'spaceAfter': 20,
                'leftIndent': 10
            },
            'ModernHeading2': {
                'parent': self.styles['Heading2'],
                'fontSize': 14,
                'textColor': colors.HexColor('#A23B72'),
                'spaceAfter': 12
            },
            'InsightBox': {
                'parent': self.styles['Normal'],
                'backColor': colors.HexColor('#ECF0F1'),
                'borderColor': colors.HexColor('#2E86AB'),
                'borderWidth': 1,
                'borderPadding': 10,
                'leftIndent': 10,
                'rightIndent': 10
            },
            'Recommendation': {
                'parent': self.styles['Normal'],
                'backColor': colors.HexColor('#E8F5E8'),
                'borderColor': colors.HexColor('#2E8B57'),
                'borderWidth': 1,
                'borderPadding': 8,
                'leftIndent': 8
            }
        }
        
        for style_name, style_config in custom_styles.items():
            if not hasattr(self.styles, style_name):
                self.styles.add(ParagraphStyle(name=style_name, **style_config))
    
    def generate_report(self, analysis_data: Dict[str, Any], output_path: str):
        """Generate modern PDF report with real plots and charts"""
        print(f"Generating modern PDF report with plots: {output_path}")
        
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=0.5*inch)
            story = []
            
            # Cover page
            story.extend(self._create_modern_cover(analysis_data))
            story.append(self._page_break())
            
            # Executive Summary with Insights
            story.extend(self._create_executive_summary(analysis_data))
            story.append(self._page_break())
            
            # Advanced Analytics with REAL plots
            story.extend(self._create_advanced_analytics(analysis_data))
            story.append(self._page_break())
            
            # Actionable Insights & Recommendations
            story.extend(self._create_actionable_insights(analysis_data))
            story.append(self._page_break())
            
            # Detailed Analysis
            story.extend(self._create_detailed_analysis(analysis_data))
            
            doc.build(story)
            print(f"Modern PDF report with plots generated successfully: {output_path}")
            
        except Exception as e:
            print(f"Error generating modern PDF report: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to simple report
            self._generate_fallback_report(analysis_data, output_path)
        finally:
            # Clean up temporary chart files
            self._cleanup_temp_files()
    
    def _create_modern_cover(self, data: Dict[str, Any]) -> List[Any]:
        """Create modern cover page with sentiment gauge"""
        elements = []
        
        # Title with modern styling
        title = Paragraph("SENTIMENTSCOPE AI ANALYSIS", self.styles['ModernTitle'])
        elements.append(title)
        elements.append(Spacer(1, 0.3*inch))
        
        # File info
        processed_date = datetime.fromisoformat(data['processed_at'].replace('Z', '+00:00'))
        file_info = f"""
        <b>Document:</b> {data['file_name']}<br/>
        <b>Analysis Date:</b> {processed_date.strftime('%B %d, %Y at %H:%M')}<br/>
        <b>Total Pages:</b> {data['document_stats']['total_pages']} | <b>Sections Analyzed:</b> {data['document_stats']['total_sections']}
        """
        elements.append(Paragraph(file_info, self.styles['ModernHeading2']))
        elements.append(Spacer(1, 0.4*inch))
        
        # Create sentiment gauge chart
        sentiment_gauge = self._create_sentiment_gauge(data)
        if sentiment_gauge:
            elements.append(Paragraph("Overall Sentiment Score", self.styles['ModernHeading2']))
            elements.append(Image(sentiment_gauge, width=5*inch, height=3*inch))
            elements.append(Spacer(1, 0.2*inch))
        
        # Key metrics table
        stats = data['document_stats']
        metrics_data = [
            ["KEY METRICS", "VALUE", "INTERPRETATION"],
            ["Dominant Emotion", stats['dominant_emotion'], self._get_emotion_insight(stats['dominant_emotion'])],
            ["Average Confidence", f"{stats['average_confidence']:.1%}", "High Reliability" if stats['average_confidence'] > 0.7 else "Moderate Reliability"],
            ["Analysis Model", list(stats.get('model_usage', {}).keys())[0] if stats.get('model_usage') else "AI-Powered", "Advanced AI Analysis"]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
        ]))
        
        elements.append(metrics_table)
        return elements
    
    def _create_executive_summary(self, data: Dict[str, Any]) -> List[Any]:
        """Create executive summary with emotion distribution chart"""
        elements = []
        
        elements.append(Paragraph("Executive Summary & Key Insights", self.styles['ModernHeading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Generate AI-powered insights
        insights = self._generate_ai_insights(data)
        
        # Overall assessment
        overall_text = f"""
        This comprehensive sentiment analysis reveals a document with <b>{insights['overall_tone']}</b> tone. 
        The analysis achieved <b>{data['document_stats']['average_confidence']:.1%} average confidence</b> across {data['document_stats']['total_sections']} sections, 
        indicating {'high reliability' if data['document_stats']['average_confidence'] > 0.7 else 'moderate reliability'} in the findings.
        """
        elements.append(Paragraph(overall_text, self.styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Emotion distribution pie chart
        emotion_chart = self._create_emotion_pie_chart(data['analytics']['emotion_distribution'])
        if emotion_chart:
            elements.append(Paragraph("Emotion Distribution", self.styles['ModernHeading2']))
            elements.append(Image(emotion_chart, width=6*inch, height=4*inch))
            elements.append(Spacer(1, 0.2*inch))
        
        # Key insights
        elements.append(Paragraph("Key Emotional Insights", self.styles['ModernHeading2']))
        for insight in insights['key_insights'][:3]:
            insight_text = f"• {insight}"
            elements.append(Paragraph(insight_text, self.styles['InsightBox']))
            elements.append(Spacer(1, 0.1*inch))
        
        return elements
    
    def _create_advanced_analytics(self, data: Dict[str, Any]) -> List[Any]:
        """Create advanced analytics section with multiple real plots"""
        elements = []
        
        elements.append(Paragraph("Advanced Analytics Dashboard", self.styles['ModernHeading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Create analytics dashboard with multiple charts
        dashboard_chart = self._create_analytics_dashboard(data)
        if dashboard_chart:
            elements.append(Image(dashboard_chart, width=6*inch, height=6*inch))
            elements.append(Spacer(1, 0.2*inch))
        
        # Intent analysis bar chart
        intent_chart = self._create_intent_bar_chart(data['analytics']['intent_distribution'])
        if intent_chart:
            elements.append(Paragraph("Communication Intent Analysis", self.styles['ModernHeading2']))
            elements.append(Image(intent_chart, width=6*inch, height=4*inch))
            elements.append(Spacer(1, 0.2*inch))
        
        # Confidence distribution
        confidence_chart = self._create_confidence_histogram(data)
        if confidence_chart:
            elements.append(Paragraph("Confidence Distribution Analysis", self.styles['ModernHeading2']))
            elements.append(Image(confidence_chart, width=6*inch, height=4*inch))
            elements.append(Spacer(1, 0.2*inch))
        
        # Emotion timeline
        timeline_chart = self._create_emotion_timeline(data)
        if timeline_chart:
            elements.append(Paragraph("Emotion Timeline Across Document", self.styles['ModernHeading2']))
            elements.append(Image(timeline_chart, width=6*inch, height=4*inch))
        
        return elements
    
    def _create_actionable_insights(self, data: Dict[str, Any]) -> List[Any]:
        """Create actionable insights and recommendations"""
        elements = []
        
        elements.append(Paragraph("Actionable Insights & Recommendations", self.styles['ModernHeading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        insights = self._generate_ai_insights(data)
        
        # Strategic recommendations
        elements.append(Paragraph("Strategic Recommendations", self.styles['ModernHeading2']))
        for i, recommendation in enumerate(insights['recommendations'][:6], 1):
            rec_text = f"<b>Recommendation {i}:</b> {recommendation}"
            elements.append(Paragraph(rec_text, self.styles['Recommendation']))
            elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Spacer(1, 0.2*inch))
        
        # Risk assessment
        elements.append(Paragraph("Risk Assessment & Considerations", self.styles['ModernHeading2']))
        risk_factors = self._assess_risks(data)
        for risk in risk_factors:
            risk_text = f"• {risk}"
            elements.append(Paragraph(risk_text, self.styles['Normal']))
            elements.append(Spacer(1, 0.05*inch))
        
        elements.append(Spacer(1, 0.2*inch))
        
        # Decision support matrix
        decision_chart = self._create_decision_matrix(data)
        if decision_chart:
            elements.append(Paragraph("Decision Support Matrix", self.styles['ModernHeading2']))
            elements.append(Image(decision_chart, width=6*inch, height=4*inch))
        
        return elements
    
    def _create_detailed_analysis(self, data: Dict[str, Any]) -> List[Any]:
        """Create detailed analysis section"""
        elements = []
        
        elements.append(Paragraph("Detailed Section Analysis", self.styles['ModernHeading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Create a modern table with analysis results
        detailed_data = [["Page", "Emotion", "Intent", "Confidence", "Text Preview"]]
        
        for result in data['analysis'][:8]:  # Show first 8 for readability
            emotion_icon = self._get_emotion_icon(result['emotion'])
            detailed_data.append([
                str(result['page']),
                f"{emotion_icon} {result['emotion']}",
                result['intent'],
                f"{result.get('confidence', 0):.0%}",
                result.get('text_snippet', '')[:35] + "..."
            ])
        
        detailed_table = Table(detailed_data, colWidths=[0.5*inch, 1.2*inch, 1.2*inch, 1*inch, 2*inch])
        detailed_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
        ]))
        
        elements.append(detailed_table)
        
        if len(data['analysis']) > 8:
            elements.append(Spacer(1, 0.1*inch))
            elements.append(Paragraph(f"... and {len(data['analysis']) - 8} more sections analyzed", self.styles['Normal']))
        
        return elements
    
    def _create_sentiment_gauge(self, data: Dict[str, Any]) -> str:
        """Create sentiment gauge chart"""
        try:
            fig, ax = plt.subplots(figsize=(8, 4), subplot_kw=dict(projection='polar'))
            
            # Calculate overall sentiment score
            emotion_dist = data['analytics']['emotion_distribution']
            sentiment_weights = {
                'Optimistic': 1.0, 'Excited': 0.8, 'Neutral': 0.5,
                'Cautious': 0.3, 'Fearful': -0.2, 'Sad': -0.5, 'Angry': -0.8
            }
            
            total_score = sum(emotion_dist.get(emotion, 0) * weight 
                            for emotion, weight in sentiment_weights.items())
            normalized_score = (total_score + 1) / 2  # Normalize to 0-1
            
            # Create gauge
            theta = np.linspace(0, np.pi, 100)
            radii = np.ones(100) * 0.8
            
            # Color segments
            colors_gauge = ['#FF6B6B', '#FFD166', '#2E8B57', '#2E8B57']
            theta_segments = np.linspace(0, np.pi, 5)
            
            for i in range(4):
                theta_start = theta_segments[i]
                theta_end = theta_segments[i+1]
                theta_range = np.linspace(theta_start, theta_end, 25)
                ax.fill_between(theta_range, 0, 0.8, color=colors_gauge[i], alpha=0.6)
            
            # Needle
            needle_angle = normalized_score * np.pi
            ax.arrow(needle_angle, 0, 0, 0.7, head_width=0.1, head_length=0.1, 
                    fc='#2C3E50', ec='#2C3E50', linewidth=3)
            
            # Labels
            ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
            ax.set_xticklabels(['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'])
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.grid(True, alpha=0.3)
            
            # Title and score
            ax.set_title(f'Overall Sentiment Score: {normalized_score:.2f}', pad=20, fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            chart_path = self._save_chart(fig, "sentiment_gauge")
            return chart_path
            
        except Exception as e:
            print(f"Sentiment gauge failed: {e}")
            return None
    
    def _create_emotion_pie_chart(self, emotion_dist: Dict[str, float]) -> str:
        """Create emotion distribution pie chart"""
        try:
            if not emotion_dist:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            emotions = list(emotion_dist.keys())
            values = [emotion_dist[e] for e in emotions]
            colors = self.colors[:len(emotions)]
            
            # Create pie chart with modern styling
            wedges, texts, autotexts = ax.pie(values, labels=emotions, colors=colors, autopct='%1.1f%%',
                                            startangle=90, shadow=True, explode=[0.05] * len(emotions))
            
            # Style the text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
            
            for text in texts:
                text.set_fontsize(11)
                text.set_fontweight('bold')
            
            ax.set_title('Emotion Distribution Analysis', fontsize=16, fontweight='bold', pad=20)
            ax.axis('equal')  # Equal aspect ratio ensures pie is circular
            
            plt.tight_layout()
            chart_path = self._save_chart(fig, "emotion_pie")
            return chart_path
            
        except Exception as e:
            print(f"Emotion pie chart failed: {e}")
            return None
    
    def _create_analytics_dashboard(self, data: Dict[str, Any]) -> str:
        """Create comprehensive analytics dashboard"""
        try:
            fig = plt.figure(figsize=(12, 10))
            
            # Create 2x2 grid
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # 1. Emotion bar chart
            ax1 = fig.add_subplot(gs[0, 0])
            emotion_dist = data['analytics']['emotion_distribution']
            emotions = list(emotion_dist.keys())
            values = [emotion_dist[e] for e in emotions]
            
            bars = ax1.bar(emotions, values, color=self.colors[:len(emotions)], alpha=0.8)
            ax1.set_title('Emotion Distribution', fontweight='bold')
            ax1.set_ylabel('Percentage')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
            
            # 2. Intent bar chart
            ax2 = fig.add_subplot(gs[0, 1])
            intent_dist = data['analytics']['intent_distribution']
            intents = list(intent_dist.keys())
            intent_values = [intent_dist[i] for i in intents]
            
            bars2 = ax2.bar(intents, intent_values, color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8)
            ax2.set_title('Intent Distribution', fontweight='bold')
            ax2.set_ylabel('Percentage')
            
            for bar, value in zip(bars2, intent_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
            
            # 3. Confidence histogram
            ax3 = fig.add_subplot(gs[1, 0])
            confidences = [r.get('confidence', 0.5) for r in data['analysis']]
            ax3.hist(confidences, bins=10, color='#2E8B57', alpha=0.7, edgecolor='black')
            ax3.set_title('Confidence Distribution', fontweight='bold')
            ax3.set_xlabel('Confidence Score')
            ax3.set_ylabel('Frequency')
            ax3.axvline(np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.2f}')
            ax3.legend()
            
            # 4. Page-wise sentiment
            ax4 = fig.add_subplot(gs[1, 1])
            pages = [r['page'] for r in data['analysis']]
            sentiment_scores = [self._emotion_to_number(r['emotion']) for r in data['analysis']]
            
            ax4.scatter(pages, sentiment_scores, alpha=0.6, color='#FF6B6B', s=50)
            
            # Add trend line
            if len(pages) > 1:
                z = np.polyfit(pages, sentiment_scores, 1)
                p = np.poly1d(z)
                ax4.plot(pages, p(pages), "r--", alpha=0.8, linewidth=2)
            
            ax4.set_title('Sentiment Trend by Page', fontweight='bold')
            ax4.set_xlabel('Page Number')
            ax4.set_ylabel('Sentiment Score')
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle('Comprehensive Sentiment Analysis Dashboard', fontsize=16, fontweight='bold', y=0.95)
            plt.tight_layout()
            
            chart_path = self._save_chart(fig, "analytics_dashboard")
            return chart_path
            
        except Exception as e:
            print(f"Analytics dashboard failed: {e}")
            return None
    
    def _create_intent_bar_chart(self, intent_dist: Dict[str, float]) -> str:
        """Create intent analysis bar chart"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            intents = list(intent_dist.keys())
            values = [intent_dist[i] for i in intents]
            colors = ['#2E86AB', '#A23B72', '#F18F01']
            
            bars = ax.bar(intents, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels and styling
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
            
            ax.set_ylabel('Percentage', fontweight='bold')
            ax.set_title('Communication Intent Analysis', fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            chart_path = self._save_chart(fig, "intent_analysis")
            return chart_path
            
        except Exception as e:
            print(f"Intent bar chart failed: {e}")
            return None
    
    def _create_confidence_histogram(self, data: Dict[str, Any]) -> str:
        """Create confidence distribution histogram"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            confidences = [r.get('confidence', 0.5) for r in data['analysis']]
            
            # Create histogram with KDE
            n, bins, patches = ax.hist(confidences, bins=12, color='#2E8B57', alpha=0.7, 
                                     edgecolor='black', density=True)
            
            # Add KDE line
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(confidences)
            x_range = np.linspace(0, 1, 100)
            ax.plot(x_range, kde(x_range), color='#2C3E50', linewidth=2, label='Density')
            
            # Add statistics
            mean_conf = np.mean(confidences)
            std_conf = np.std(confidences)
            
            ax.axvline(mean_conf, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_conf:.2f}')
            ax.axvline(mean_conf + std_conf, color='orange', linestyle=':', alpha=0.7)
            ax.axvline(mean_conf - std_conf, color='orange', linestyle=':', alpha=0.7)
            
            ax.set_xlabel('Confidence Score', fontweight='bold')
            ax.set_ylabel('Density', fontweight='bold')
            ax.set_title('Confidence Distribution Analysis', fontsize=14, fontweight='bold', pad=20)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add text box with statistics
            stats_text = f'Statistics:\nMean: {mean_conf:.3f}\nStd: {std_conf:.3f}\nN: {len(confidences)}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontfamily='monospace')
            
            plt.tight_layout()
            chart_path = self._save_chart(fig, "confidence_histogram")
            return chart_path
            
        except Exception as e:
            print(f"Confidence histogram failed: {e}")
            return None
    
    def _create_emotion_timeline(self, data: Dict[str, Any]) -> str:
        """Create emotion timeline across pages"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Group by page and calculate average sentiment
            page_sentiments = {}
            for result in data['analysis']:
                page = result['page']
                if page not in page_sentiments:
                    page_sentiments[page] = []
                page_sentiments[page].append(self._emotion_to_number(result['emotion']))
            
            pages = sorted(page_sentiments.keys())
            avg_sentiments = [np.mean(page_sentiments[page]) for page in pages]
            
            # Create line plot with confidence interval
            ax.plot(pages, avg_sentiments, marker='o', linewidth=3, markersize=8, 
                   color='#2E86AB', label='Average Sentiment')
            
            # Add confidence interval
            std_sentiments = [np.std(page_sentiments[page]) for page in pages]
            ax.fill_between(pages, 
                          [avg - std for avg, std in zip(avg_sentiments, std_sentiments)],
                          [avg + std for avg, std in zip(avg_sentiments, std_sentiments)],
                          alpha=0.2, color='#2E86AB', label='±1 Std Dev')
            
            ax.set_xlabel('Page Number', fontweight='bold')
            ax.set_ylabel('Sentiment Score', fontweight='bold')
            ax.set_title('Emotion Timeline Across Document Pages', fontsize=14, fontweight='bold', pad=20)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add horizontal line at y=0
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            chart_path = self._save_chart(fig, "emotion_timeline")
            return chart_path
            
        except Exception as e:
            print(f"Emotion timeline failed: {e}")
            return None
    
    def _create_decision_matrix(self, data: Dict[str, Any]) -> str:
        """Create decision support matrix"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extract key metrics for decision matrix
            emotion_dist = data['analytics']['emotion_distribution']
            intent_dist = data['analytics']['intent_distribution']
            avg_confidence = data['document_stats']['average_confidence']
            
            # Calculate scores for decision matrix
            positivity_score = emotion_dist.get('Optimistic', 0) + emotion_dist.get('Excited', 0)
            caution_score = emotion_dist.get('Fearful', 0) + emotion_dist.get('Cautious', 0)
            persuasion_score = intent_dist.get('Persuasive', 0)
            clarity_score = avg_confidence
            
            metrics = ['Positivity', 'Caution', 'Persuasion', 'Clarity']
            scores = [positivity_score, caution_score, persuasion_score, clarity_score]
            colors = ['#2E8B57', '#FFA500', '#A23B72', '#2E86AB']
            
            # Create radar chart
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            scores += scores[:1]  # Complete the circle
            angles += angles[:1]  # Complete the circle
            
            ax = fig.add_subplot(111, polar=True)
            ax.plot(angles, scores, 'o-', linewidth=2, markersize=8, color='#2E86AB')
            ax.fill(angles, scores, alpha=0.25, color='#2E86AB')
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
            ax.grid(True)
            
            ax.set_title('Decision Support Matrix\nDocument Communication Profile', 
                        fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            chart_path = self._save_chart(fig, "decision_matrix")
            return chart_path
            
        except Exception as e:
            print(f"Decision matrix failed: {e}")
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
            insights['key_insights'].append("Strong positive sentiment detected - excellent for stakeholder engagement")
        if emotion_dist.get('Fearful', 0) > 0.3:
            insights['key_insights'].append("Significant cautious tones - consider proactive risk communication")
        if emotion_dist.get('Angry', 0) > 0.2:
            insights['key_insights'].append("Negative emotional tones present - may require immediate attention")
        
        if intent_dist.get('Persuasive', 0) > 0.5:
            insights['key_insights'].append("Highly persuasive document - effective call-to-action strategy")
        elif intent_dist.get('Informative', 0) > 0.6:
            insights['key_insights'].append("Primarily informative - strong factual foundation")
        
        # Generate recommendations
        recommendations = {
            'Optimistic': [
                "Capitalize on positive momentum in stakeholder communications",
                "Highlight successful strategies and achievements",
                "Use optimistic tone in follow-up messaging",
                "Consider expanding on positive themes in future content"
            ],
            'Fearful': [
                "Address uncertainties with clear, factual information",
                "Develop comprehensive risk mitigation strategies",
                "Provide reassurance through transparent communication",
                "Consider additional data to support decision-making"
            ],
            'Angry': [
                "Immediately address underlying concerns or issues",
                "Review and adjust communication tone if necessary",
                "Implement structured feedback collection",
                "Develop action plan for issue resolution"
            ],
            'Neutral': [
                "Maintain factual and balanced communication approach",
                "Consider adding emotional appeal for better engagement",
                "Ensure clarity and precision in messaging",
                "Monitor audience reception for tone appropriateness"
            ]
        }
        
        # Add emotion-specific recommendations
        insights['recommendations'].extend(recommendations.get(dominant_emotion, [
            "Maintain current communication strategy",
            "Continue monitoring emotional tone consistency",
            "Consider audience-specific tone adjustments"
        ]))
        
        # Add confidence-based recommendations
        if avg_confidence < 0.6:
            insights['recommendations'].append("Review low-confidence sections for clarity improvement")
        
        # Add intent-based recommendations
        if intent_dist.get('Persuasive', 0) > 0.4:
            insights['recommendations'].append("Strengthen persuasive elements with data-driven evidence")
        
        return insights
    
    def _assess_risks(self, data: Dict[str, Any]) -> List[str]:
        """Assess potential risks based on analysis"""
        risks = []
        emotion_dist = data['analytics']['emotion_distribution']
        avg_confidence = data['document_stats']['average_confidence']
        
        if emotion_dist.get('Fearful', 0) > 0.3:
            risks.append("High uncertainty may affect stakeholder confidence and decision-making")
        if emotion_dist.get('Angry', 0) > 0.2:
            risks.append("Negative tones could impact document reception and effectiveness")
        if avg_confidence < 0.6:
            risks.append("Ambiguous content in low-confidence sections may require clarification")
        if emotion_dist.get('Neutral', 0) > 0.7:
            risks.append("Highly neutral tone may reduce emotional engagement and memorability")
        
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
            return "predominantly cautious with notable concerns"
        elif emotion_dist.get('Neutral', 0) > 0.6:
            return "balanced, factual and professionally neutral"
        else:
            return "mixed with varied emotional tones reflecting complexity"
    
    def _emotion_to_number(self, emotion: str) -> float:
        """Convert emotion to numerical score"""
        emotion_scores = {
            'Angry': -1.0, 'Sad': -0.7, 'Fearful': -0.3,
            'Neutral': 0.0, 'Cautious': 0.1, 'Optimistic': 0.7, 'Excited': 1.0
        }
        return emotion_scores.get(emotion, 0.0)
    
    def _get_emotion_insight(self, emotion: str) -> str:
        """Get insight for specific emotion"""
        insights = {
            'Optimistic': "Positive and forward-looking - excellent for engagement",
            'Fearful': "Cautious tone - indicates areas needing attention", 
            'Angry': "Strong negative sentiment - requires immediate addressing",
            'Sad': "Somber tone - may need supportive communication",
            'Neutral': "Balanced and factual - professional communication",
            'Excited': "Enthusiastic - high energy and positive momentum", 
            'Cautious': "Careful approach - risk-aware and measured"
        }
        return insights.get(emotion, "Appropriate emotional tone for context")
    
    def _get_emotion_icon(self, emotion: str) -> str:
        """Get emotion icon (using text representation)"""
        icons = {
            'Optimistic': "↑", 'Fearful': "⚠", 'Angry': "⚡",
            'Sad': "↓", 'Neutral': "•", 'Excited': "★", 'Cautious': "ⓘ"
        }
        return icons.get(emotion, "•")
    
    def _save_chart(self, fig, name: str) -> str:
        """Save matplotlib chart as image and return file path"""
        try:
            chart_path = f"temp_{name}.png"
            fig.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)  # Close the figure to free memory
            self.chart_temp_files.append(chart_path)
            return chart_path
        except Exception as e:
            print(f"Failed to save chart {name}: {e}")
            plt.close(fig)
            return None
    
    def _page_break(self) -> Spacer:
        """Create a page break"""
        return Spacer(1, 0.1*inch)
    
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