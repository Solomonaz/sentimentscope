#!/usr/bin/env python3
"""
Sample PDF Generator for SentimentScope Testing
Creates a PDF with varied emotional tones and persuasive intents
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import os

def create_sample_pdf():
    """Create a sample PDF with varied emotional content"""
    
    # Create input_pdfs directory if it doesn't exist
    os.makedirs('input_pdfs', exist_ok=True)
    
    # Create PDF document
    pdf_path = 'input_pdfs/sample_business_report.pdf'
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=1*inch)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        name='Title',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        textColor=colors.HexColor('#2C3E50')
    )
    
    heading_style = ParagraphStyle(
        name='Heading2',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.HexColor('#34495E')
    )
    
    normal_style = styles['BodyText']
    
    # Cover Page - Optimistic tone
    story.append(Paragraph("Quarterly Business Performance Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Q4 2024 Analysis", heading_style))
    story.append(Spacer(1, 0.5*inch))
    
    optimistic_text = """
    We are thrilled to announce exceptional growth this quarter, with revenue increasing by 23% year-over-year. 
    Our strategic investments in innovation are paying remarkable dividends, and we see tremendous opportunities 
    ahead in the evolving market landscape. The future looks incredibly bright for our organization and stakeholders.
    """
    story.append(Paragraph(optimistic_text, normal_style))
    story.append(PageBreak())
    
    # Page 2 - Executive Summary (Mixed emotions)
    story.append(Paragraph("Executive Summary", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    mixed_text = """
    This quarter has been transformative for our company. While we celebrate outstanding performance in several key areas, 
    we must also address emerging challenges in the supply chain. The global economic situation requires careful navigation, 
    but we remain confident in our team's ability to adapt and excel.
    
    We strongly recommend increasing our investment in digital transformation initiatives. The data clearly shows that 
    companies embracing AI and automation are outperforming their peers. Our analysis indicates a potential 15-20% 
    efficiency gain through targeted technological upgrades.
    """
    story.append(Paragraph(mixed_text, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Page 3 - Challenges Section (Fearful/Cautious)
    story.append(Paragraph("Market Challenges & Risk Assessment", heading_style))
    
    cautious_text = """
    We are deeply concerned about the increasing volatility in raw material prices. Recent geopolitical developments 
    have created significant uncertainty in our primary supply markets. There is a real risk of cost escalation that 
    could impact our profit margins if current trends continue.
    
    We must carefully monitor these developments and consider diversifying our supplier base. The potential for 
    disruption is substantial, and contingency planning should be prioritized. We advise implementing additional 
    risk mitigation strategies immediately.
    """
    story.append(Paragraph(cautious_text, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Page 4 - Financial Performance (Neutral/Informative)
    story.append(Paragraph("Financial Performance Analysis", heading_style))
    
    neutral_text = """
    According to the financial data, revenue reached $45.2 million this quarter, representing a 15% increase 
    over the previous quarter. Operating expenses totaled $28.7 million, resulting in a net profit margin of 18.3%. 
    The data indicates consistent growth across all major product lines.
    
    Research findings show that customer acquisition costs have decreased by 7% while customer lifetime value 
    has increased by 12%. These metrics suggest improved marketing efficiency and stronger customer relationships.
    """
    story.append(Paragraph(neutral_text, normal_style))
    story.append(PageBreak())
    
    # Page 5 - Strategic Recommendations (Persuasive/Excited)
    story.append(Paragraph("Strategic Recommendations", heading_style))
    
    persuasive_text = """
    We must aggressively pursue expansion into Asian markets. The opportunity is massive and timing is critical. 
    Our analysis reveals untapped potential that could double our market share within 18 months. We absolutely need 
    to act now before competitors establish dominant positions.
    
    The innovation team has developed breakthrough technology that will revolutionize our industry. This is an 
    amazing opportunity to establish market leadership. We should immediately allocate resources to accelerate 
    development and launch. The potential returns are extraordinary!
    """
    story.append(Paragraph(persuasive_text, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Page 6 - Operational Issues (Angry/Sad)
    story.append(Paragraph("Operational Challenges", heading_style))
    
    negative_text = """
    The recent system failure was completely unacceptable and reflects serious shortcomings in our IT infrastructure. 
    This failure caused significant disruption to our operations and damaged client relationships. We cannot tolerate 
    such failures in the future.
    
    Unfortunately, we experienced a disappointing 8% decline in customer satisfaction scores this quarter. This 
    decline is deeply concerning and reflects underlying issues in our service delivery. We regret these shortcomings 
    and are committed to addressing them immediately.
    """
    story.append(Paragraph(negative_text, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Page 7 - Conclusion (Mixed with strong persuasive intent)
    story.append(Paragraph("Conclusion & Call to Action", heading_style))
    
    conclusion_text = """
    In conclusion, while we face some challenges, the overall outlook remains strongly positive. We have 
    demonstrated resilience and innovation throughout this quarter. The data unequivocally supports continued 
    investment in our growth strategy.
    
    We urgently need to implement the recommendations outlined in this report. Delaying action would be a 
    serious mistake that could compromise our competitive position. The board must approve the proposed 
    initiatives without hesitation to capitalize on current market opportunities.
    
    Our organization stands at a pivotal moment. With decisive action and continued innovation, we will 
    achieve unprecedented success in the coming year. The time to act is now!
    """
    story.append(Paragraph(conclusion_text, normal_style))
    
    # Build PDF
    doc.build(story)
    print(f"✅ Sample PDF created: {pdf_path}")
    print("📄 This PDF contains varied emotional tones perfect for testing SentimentScope:")
    print("   - Page 1: Optimistic/Persuasive")
    print("   - Page 2: Mixed emotions with persuasive recommendations")
    print("   - Page 3: Fearful/Cautious with advisory tone")
    print("   - Page 4: Neutral/Informative with data focus")
    print("   - Page 5: Excited/Persuasive with strong recommendations")
    print("   - Page 6: Angry/Sad addressing problems")
    print("   - Page 7: Mixed with strong persuasive conclusion")

if __name__ == "__main__":
    create_sample_pdf()