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
    """Modern PDF report generator with BOTH visual charts AND detailed tables"""
    
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
            },
            'DataTable': {
                'parent': self.styles['Normal'],
                'fontSize': 8,
                'leading': 10
            }
        }
        
        for style_name, style_config in custom_styles.items():
            if not hasattr(self.styles, style_name):
                self.styles.add(ParagraphStyle(name=style_name, **style_config))
    
    def generate_report(self, analysis_data: Dict[str, Any], output_path: str):
        """Generate modern PDF report with BOTH charts and tables"""
        print(f"Generating comprehensive PDF report: {output_path}")
        
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=0.5*inch)
            story = []
            
            # Cover page
            story.extend(self._create_modern_cover(analysis_data))
            story.append(self._page_break())
            
            # Executive Summary with Insights
            story.extend(self._create_executive_summary(analysis_data))
            story.append(self._page_break())
            
            # Comprehensive Analytics - BOTH Visual and Tabular
            story.extend(self._create_comprehensive_analytics(analysis_data))
            story.append(self._page_break())
            
            # Actionable Insights & Recommendations
            story.extend(self._create_actionable_insights(analysis_data))
            story.append(self._page_break())
            
            # Detailed Analysis
            story.extend(self._create_detailed_analysis(analysis_data))
            
            doc.build(story)
            print(f"Comprehensive PDF report generated successfully: {output_path}")
            
        except Exception as e:
            print(f"Error generating comprehensive PDF report: {e}")
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
            ["Total Pages", str(stats['total_pages']), "Document Length"],
            ["Sections Analyzed", str(stats['total_sections']), "Content Coverage"],
            ["Processing Time", f"{stats.get('processing_time_seconds', 0):.1f}s", "Analysis Duration"]
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
        """Create executive summary with emotion distribution"""
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
        
        # Emotion distribution - BOTH chart and table
        emotion_chart = self._create_emotion_pie_chart(data['analytics']['emotion_distribution'])
        if emotion_chart:
            elements.append(Paragraph("Emotion Distribution Analysis", self.styles['ModernHeading2']))
            elements.append(Image(emotion_chart, width=6*inch, height=4*inch))
            elements.append(Spacer(1, 0.2*inch))
        
        # Emotion distribution table
        elements.append(Paragraph("Emotion Distribution - Detailed Table", self.styles['ModernHeading2']))
        emotion_table = self._create_emotion_distribution_table(data['analytics']['emotion_distribution'])
        elements.append(emotion_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Key insights
        elements.append(Paragraph("Key Emotional Insights", self.styles['ModernHeading2']))
        for insight in insights['key_insights'][:3]:
            insight_text = f"• {insight}"
            elements.append(Paragraph(insight_text, self.styles['InsightBox']))
            elements.append(Spacer(1, 0.1*inch))
        
        return elements
    
    def _create_comprehensive_analytics(self, data: Dict[str, Any]) -> List[Any]:
        """Create comprehensive analytics section with BOTH charts and tables"""
        elements = []
        
        elements.append(Paragraph("Comprehensive Analytics Dashboard", self.styles['ModernHeading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        # 1. Advanced Analytics Dashboard Chart
        dashboard_chart = self._create_analytics_dashboard(data)
        if dashboard_chart:
            elements.append(Paragraph("Advanced Analytics Dashboard - Visual Overview", self.styles['ModernHeading2']))
            elements.append(Image(dashboard_chart, width=6*inch, height=6*inch))
            elements.append(Spacer(1, 0.2*inch))
        
        # 2. Emotion Distribution - Detailed Table
        elements.append(Paragraph("Emotion Distribution - Statistical Analysis", self.styles['ModernHeading2']))
        emotion_stats_table = self._create_emotion_statistical_table(data['analytics']['emotion_distribution'])
        elements.append(emotion_stats_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # 3. Communication Intent Analysis - BOTH
        intent_chart = self._create_intent_bar_chart(data['analytics']['intent_distribution'])
        if intent_chart:
            elements.append(Paragraph("Communication Intent Analysis - Visual", self.styles['ModernHeading2']))
            elements.append(Image(intent_chart, width=6*inch, height=4*inch))
            elements.append(Spacer(1, 0.2*inch))
        
        # Intent Analysis Table
        elements.append(Paragraph("Communication Intent - Detailed Analysis", self.styles['ModernHeading2']))
        intent_table = self._create_intent_analysis_table(data['analytics']['intent_distribution'])
        elements.append(intent_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # 4. Confidence Distribution - BOTH
        confidence_chart = self._create_confidence_histogram(data)
        if confidence_chart:
            elements.append(Paragraph("Confidence Distribution Analysis - Visual", self.styles['ModernHeading2']))
            elements.append(Image(confidence_chart, width=6*inch, height=4*inch))
            elements.append(Spacer(1, 0.2*inch))
        
        # Confidence Analysis Table
        elements.append(Paragraph("Confidence Distribution - Statistical Summary", self.styles['ModernHeading2']))
        confidence_table = self._create_confidence_statistical_table(data)
        elements.append(confidence_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # 5. Emotion Timeline - BOTH
        timeline_chart = self._create_emotion_timeline(data)
        if timeline_chart:
            elements.append(Paragraph("Emotion Timeline Across Document - Visual", self.styles['ModernHeading2']))
            elements.append(Image(timeline_chart, width=6*inch, height=4*inch))
            elements.append(Spacer(1, 0.2*inch))
        
        # Timeline Analysis Table
        elements.append(Paragraph("Emotion Timeline - Page-wise Analysis", self.styles['ModernHeading2']))
        timeline_table = self._create_timeline_analysis_table(data)
        elements.append(timeline_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # 6. Decision Support Matrix - BOTH
        decision_chart = self._create_decision_matrix(data)
        if decision_chart:
            elements.append(Paragraph("Decision Support Matrix - Visual", self.styles['ModernHeading2']))
            elements.append(Image(decision_chart, width=6*inch, height=4*inch))
            elements.append(Spacer(1, 0.2*inch))
        
        # Decision Matrix Table
        elements.append(Paragraph("Decision Support Matrix - Quantitative Analysis", self.styles['ModernHeading2']))
        decision_table = self._create_decision_support_table(data)
        elements.append(decision_table)
        
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
        
        return elements
    
    def _create_detailed_analysis(self, data: Dict[str, Any]) -> List[Any]:
        """Create detailed analysis section"""
        elements = []
        
        elements.append(Paragraph("Detailed Section Analysis", self.styles['ModernHeading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Create a comprehensive table with analysis results
        detailed_data = [["Page", "Emotion", "Intent", "Confidence", "Model Used", "Text Preview"]]
        
        for result in data['analysis'][:15]:  # Show first 15 for readability
            emotion_icon = self._get_emotion_icon(result['emotion'])
            detailed_data.append([
                str(result['page']),
                f"{emotion_icon} {result['emotion']}",
                result['intent'],
                f"{result.get('confidence', 0):.0%}",
                result.get('model_used', 'N/A'),
                result.get('text_snippet', '')[:30] + "..."
            ])
        
        detailed_table = Table(detailed_data, colWidths=[0.5*inch, 1.2*inch, 1.2*inch, 1*inch, 1*inch, 1.5*inch])
        detailed_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
        ]))
        
        elements.append(detailed_table)
        
        if len(data['analysis']) > 15:
            elements.append(Spacer(1, 0.1*inch))
            elements.append(Paragraph(f"... and {len(data['analysis']) - 15} more sections analyzed", self.styles['Normal']))
        
        return elements

    # ============================================================================
    # VISUAL CHART METHODS (Keep all your existing chart methods)
    # ============================================================================
    
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
            
            # Create gauge (your existing gauge code)
            theta = np.linspace(0, np.pi, 100)
            radii = np.ones(100) * 0.8
            
            colors_gauge = ['#FF6B6B', '#FFD166', '#2E8B57', '#2E8B57']
            theta_segments = np.linspace(0, np.pi, 5)
            
            for i in range(4):
                theta_start = theta_segments[i]
                theta_end = theta_segments[i+1]
                theta_range = np.linspace(theta_start, theta_end, 25)
                ax.fill_between(theta_range, 0, 0.8, color=colors_gauge[i], alpha=0.6)
            
            needle_angle = normalized_score * np.pi
            ax.arrow(needle_angle, 0, 0, 0.7, head_width=0.1, head_length=0.1, 
                    fc='#2C3E50', ec='#2C3E50', linewidth=3)
            
            ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
            ax.set_xticklabels(['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'])
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.grid(True, alpha=0.3)
            
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
            
            wedges, texts, autotexts = ax.pie(values, labels=emotions, colors=colors, autopct='%1.1f%%',
                                            startangle=90, shadow=True, explode=[0.05] * len(emotions))
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
            
            for text in texts:
                text.set_fontsize(11)
                text.set_fontweight('bold')
            
            ax.set_title('Emotion Distribution Analysis', fontsize=16, fontweight='bold', pad=20)
            ax.axis('equal')
            
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
            
            n, bins, patches = ax.hist(confidences, bins=12, color='#2E8B57', alpha=0.7, 
                                     edgecolor='black', density=True)
            
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(confidences)
            x_range = np.linspace(0, 1, 100)
            ax.plot(x_range, kde(x_range), color='#2C3E50', linewidth=2, label='Density')
            
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
            
            page_sentiments = {}
            for result in data['analysis']:
                page = result['page']
                if page not in page_sentiments:
                    page_sentiments[page] = []
                page_sentiments[page].append(self._emotion_to_number(result['emotion']))
            
            pages = sorted(page_sentiments.keys())
            avg_sentiments = [np.mean(page_sentiments[page]) for page in pages]
            
            ax.plot(pages, avg_sentiments, marker='o', linewidth=3, markersize=8, 
                   color='#2E86AB', label='Average Sentiment')
            
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
            
            emotion_dist = data['analytics']['emotion_distribution']
            intent_dist = data['analytics']['intent_distribution']
            avg_confidence = data['document_stats']['average_confidence']
            
            positivity_score = emotion_dist.get('Optimistic', 0) + emotion_dist.get('Excited', 0)
            caution_score = emotion_dist.get('Fearful', 0) + emotion_dist.get('Cautious', 0)
            persuasion_score = intent_dist.get('Persuasive', 0)
            clarity_score = avg_confidence
            
            metrics = ['Positivity', 'Caution', 'Persuasion', 'Clarity']
            scores = [positivity_score, caution_score, persuasion_score, clarity_score]
            colors = ['#2E8B57', '#FFA500', '#A23B72', '#2E86AB']
            
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            scores += scores[:1]
            angles += angles[:1]
            
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

    # ============================================================================
    # NEW TABLE METHODS (Added for detailed tabular analysis)
    # ============================================================================
    
    def _create_emotion_distribution_table(self, emotion_dist: Dict[str, float]) -> Table:
        """Create detailed emotion distribution table"""
        table_data = [["Emotion", "Percentage", "Count", "Interpretation"]]
        
        for emotion, percentage in emotion_dist.items():
            count = int(percentage * 100)  # Approximate count
            interpretation = self._get_emotion_interpretation(emotion, percentage)
            table_data.append([
                emotion,
                f"{percentage:.1%}",
                f"~{count}",
                interpretation
            ])
        
        table = Table(table_data, colWidths=[1.5*inch, 1*inch, 1*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
        ]))
        
        return table
    
    def _create_emotion_statistical_table(self, emotion_dist: Dict[str, float]) -> Table:
        """Create statistical analysis table for emotions"""
        table_data = [["Metric", "Value", "Analysis"]]
        
        # Calculate statistics
        emotions = list(emotion_dist.keys())
        values = list(emotion_dist.values())
        
        dominant_emotion = max(emotion_dist.items(), key=lambda x: x[1])[0]
        dominant_percentage = emotion_dist[dominant_emotion]
        
        emotion_diversity = len(emotions)
        max_variation = max(values) - min(values) if values else 0
        
        table_data.extend([
            ["Dominant Emotion", dominant_emotion, "Primary emotional tone"],
            ["Dominance Level", f"{dominant_percentage:.1%}", 
             "Strong" if dominant_percentage > 0.4 else "Moderate" if dominant_percentage > 0.25 else "Weak"],
            ["Emotion Diversity", str(emotion_diversity), 
             "High" if emotion_diversity > 4 else "Medium" if emotion_diversity > 2 else "Low"],
            ["Max Variation", f"{max_variation:.1%}", "Emotional range across document"],
            ["Key Emotions", ", ".join(emotions[:3]), "Most prevalent emotional tones"]
        ])
        
        table = Table(table_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#A23B72')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
        ]))
        
        return table
    
    def _create_intent_analysis_table(self, intent_dist: Dict[str, float]) -> Table:
        """Create detailed intent analysis table"""
        table_data = [["Intent Type", "Percentage", "Strength", "Communication Impact"]]
        
        for intent, percentage in intent_dist.items():
            strength = "Strong" if percentage > 0.4 else "Moderate" if percentage > 0.25 else "Weak"
            impact = self._get_intent_impact(intent, percentage)
            
            table_data.append([
                intent,
                f"{percentage:.1%}",
                strength,
                impact
            ])
        
        table = Table(table_data, colWidths=[1.5*inch, 1*inch, 1*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F18F01')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
        ]))
        
        return table
    
    def _create_confidence_statistical_table(self, data: Dict[str, Any]) -> Table:
        """Create confidence distribution statistical table"""
        confidences = [r.get('confidence', 0.5) for r in data['analysis']]
        
        if not confidences:
            confidences = [0.5]
        
        table_data = [["Statistical Measure", "Value", "Interpretation"]]
        
        table_data.extend([
            ["Mean Confidence", f"{np.mean(confidences):.3f}", 
             "High Reliability" if np.mean(confidences) > 0.7 else "Moderate Reliability"],
            ["Standard Deviation", f"{np.std(confidences):.3f}", 
             "Low Variation" if np.std(confidences) < 0.2 else "High Variation"],
            ["Minimum Confidence", f"{np.min(confidences):.3f}", "Lowest confidence score"],
            ["Maximum Confidence", f"{np.max(confidences):.3f}", "Highest confidence score"],
            ["Confidence Range", f"{np.max(confidences) - np.min(confidences):.3f}", "Spread of confidence scores"],
            ["Analysis Quality", self._get_analysis_quality(confidences), "Overall reliability assessment"]
        ])
        
        table = Table(table_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E8B57')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
        ]))
        
        return table
    
    def _create_timeline_analysis_table(self, data: Dict[str, Any]) -> Table:
        """Create page-wise emotion timeline table"""
        # Group by page
        page_emotions = {}
        for result in data['analysis']:
            page = result['page']
            if page not in page_emotions:
                page_emotions[page] = []
            page_emotions[page].append(result['emotion'])
        
        table_data = [["Page", "Dominant Emotion", "Emotion Count", "Emotion Diversity", "Sentiment Score"]]
        
        for page in sorted(page_emotions.keys())[:10]:  # Show first 10 pages
            emotions = page_emotions[page]
            emotion_count = Counter(emotions)
            dominant_emotion = emotion_count.most_common(1)[0][0]
            diversity = len(emotion_count)
            sentiment_score = np.mean([self._emotion_to_number(e) for e in emotions])
            
            table_data.append([
                str(page),
                dominant_emotion,
                str(len(emotions)),
                str(diversity),
                f"{sentiment_score:.2f}"
            ])
        
        table = Table(table_data, colWidths=[0.8*inch, 1.2*inch, 1*inch, 1.2*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6A0572')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
        ]))
        
        return table
    
    def _create_decision_support_table(self, data: Dict[str, Any]) -> Table:
        """Create decision support matrix table"""
        emotion_dist = data['analytics']['emotion_distribution']
        intent_dist = data['analytics']['intent_distribution']
        avg_confidence = data['document_stats']['average_confidence']
        
        # Calculate decision factors
        positivity = emotion_dist.get('Optimistic', 0) + emotion_dist.get('Excited', 0)
        caution = emotion_dist.get('Fearful', 0) + emotion_dist.get('Cautious', 0)
        persuasion = intent_dist.get('Persuasive', 0)
        clarity = avg_confidence
        
        table_data = [["Decision Factor", "Score", "Level", "Strategic Implication"]]
        
        factors = [
            ("Positivity Index", positivity, 
             "High" if positivity > 0.5 else "Medium" if positivity > 0.3 else "Low",
             "Positive tone supports engagement and acceptance"),
            ("Caution Level", caution,
             "High" if caution > 0.3 else "Medium" if caution > 0.15 else "Low", 
             "Indicates need for careful consideration"),
            ("Persuasion Strength", persuasion,
             "Strong" if persuasion > 0.4 else "Moderate" if persuasion > 0.25 else "Weak",
             "Effectiveness of persuasive elements"),
            ("Clarity Score", clarity,
             "High" if clarity > 0.7 else "Medium" if clarity > 0.5 else "Low",
             "Reliability of analysis results"),
            ("Emotional Balance", abs(positivity - caution),
             "Balanced" if abs(positivity - caution) < 0.2 else "Skewed",
             "Overall emotional equilibrium")
        ]
        
        for factor, score, level, implication in factors:
            table_data.append([
                factor,
                f"{score:.3f}",
                level,
                implication
            ])
        
        table = Table(table_data, colWidths=[1.8*inch, 0.8*inch, 1*inch, 2.4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FF6B6B')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
        ]))
        
        return table

    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def _get_emotion_interpretation(self, emotion: str, percentage: float) -> str:
        """Get interpretation for emotion percentage"""
        interpretations = {
            'Optimistic': "Positive outlook, good for engagement",
            'Fearful': "Indicates concerns or uncertainties", 
            'Angry': "Strong negative sentiment, needs attention",
            'Sad': "Somber tone, may need supportive approach",
            'Neutral': "Balanced and factual communication",
            'Excited': "High energy and positive momentum",
            'Cautious': "Careful, risk-aware approach"
        }
        
        base = interpretations.get(emotion, "Standard emotional tone")
        strength = " (Strong)" if percentage > 0.3 else " (Moderate)" if percentage > 0.15 else " (Weak)"
        return base + strength
    
    def _get_intent_impact(self, intent: str, percentage: float) -> str:
        """Get communication impact for intent"""
        impacts = {
            'Persuasive': "Aims to influence decisions or actions",
            'Informative': "Focuses on providing facts and data", 
            'Neutral': "Maintains objective, balanced presentation"
        }
        
        base = impacts.get(intent, "Standard communication approach")
        strength = " (Primary)" if percentage > 0.4 else " (Significant)" if percentage > 0.25 else " (Secondary)"
        return base + strength
    
    def _get_analysis_quality(self, confidences: List[float]) -> str:
        """Get overall analysis quality assessment"""
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)
        
        if mean_conf > 0.8 and std_conf < 0.15:
            return "Excellent - High confidence, consistent results"
        elif mean_conf > 0.7 and std_conf < 0.2:
            return "Good - Reliable with moderate consistency"
        elif mean_conf > 0.6:
            return "Fair - Acceptable with some variation"
        else:
            return "Needs Review - Lower confidence levels"
    
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
        
        if intent_dist.get('Persuasive', 0) > 0.5:
            insights['key_insights'].append("Highly persuasive document - effective call-to-action strategy")
        
        # Generate recommendations
        recommendations = {
            'Optimistic': [
                "Capitalize on positive momentum in stakeholder communications",
                "Highlight successful strategies and achievements",
                "Use optimistic tone in follow-up messaging"
            ],
            'Fearful': [
                "Address uncertainties with clear, factual information",
                "Develop comprehensive risk mitigation strategies",
                "Provide reassurance through transparent communication"
            ]
        }
        
        insights['recommendations'].extend(recommendations.get(dominant_emotion, [
            "Maintain current communication strategy",
            "Continue monitoring emotional tone consistency"
        ]))
        
        return insights
    
    def _assess_risks(self, data: Dict[str, Any]) -> List[str]:
        """Assess potential risks based on analysis"""
        risks = []
        emotion_dist = data['analytics']['emotion_distribution']
        avg_confidence = data['document_stats']['average_confidence']
        
        if emotion_dist.get('Fearful', 0) > 0.3:
            risks.append("High uncertainty may affect stakeholder confidence")
        if avg_confidence < 0.6:
            risks.append("Lower confidence scores may indicate ambiguous content")
        
        if not risks:
            risks.append("No significant risks identified - document maintains appropriate tone")
        
        return risks
    
    def _assess_overall_tone(self, emotion_dist: Dict[str, float]) -> str:
        """Assess overall document tone"""
        positive_score = emotion_dist.get('Optimistic', 0) + emotion_dist.get('Excited', 0)
        negative_score = emotion_dist.get('Fearful', 0) + emotion_dist.get('Angry', 0) + emotion_dist.get('Sad', 0)
        
        if positive_score > negative_score + 0.2:
            return "predominantly positive and forward-looking"
        elif negative_score > positive_score + 0.2:
            return "predominantly cautious with notable concerns"
        else:
            return "balanced with mixed emotional tones"
    
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
            'Optimistic': "Positive outlook, good for engagement",
            'Fearful': "Cautious tone, indicates areas needing attention", 
            'Angry': "Strong negative sentiment, requires immediate addressing",
            'Sad': "Somber tone, may need supportive communication",
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
            plt.close(fig)
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
            from .simple_report_builder import SimpleReportBuilder
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
                pass