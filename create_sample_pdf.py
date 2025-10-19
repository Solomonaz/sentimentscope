#!/usr/bin/env python3
"""
Test PDF Generator for SentimentScope
Creates a PDF with varied emotional tones, risks, and persuasive elements
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import os

def create_comprehensive_test_pdf():
    """Create a comprehensive test PDF with varied content"""
    
    # Create input_pdfs directory if it doesn't exist
    os.makedirs('input_pdfs', exist_ok=True)
    
    # Create PDF document
    pdf_path = 'input_pdfs/comprehensive_test_document.pdf'
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
    
    # Cover Page - Optimistic & Persuasive
    story.append(Paragraph("Strategic Business Review & Risk Assessment", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Q1 2024 Performance Analysis", heading_style))
    story.append(Spacer(1, 0.5*inch))
    
    optimistic_text = """
    We are absolutely thrilled to announce outstanding performance this quarter, with unprecedented growth 
    across all key metrics. Our strategic initiatives have yielded exceptional results, and we are 
    incredibly excited about the tremendous opportunities that lie ahead. The market response has been 
    overwhelmingly positive, and we must capitalize on this momentum immediately.
    """
    story.append(Paragraph(optimistic_text, normal_style))
    story.append(PageBreak())
    
    # Page 2 - Executive Summary (Mixed emotions with strong persuasion)
    story.append(Paragraph("Executive Summary", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    persuasive_text = """
    This quarter represents a pivotal moment for our organization. We must urgently address several 
    critical challenges while aggressively pursuing new opportunities. The board should immediately 
    approve the proposed strategic initiatives without delay. We strongly recommend increasing our 
    investment in digital transformation by 40% to maintain competitive advantage.
    
    The data unequivocally demonstrates that companies failing to adapt to market changes face 
    existential threats. We cannot afford to be complacent. Our analysis reveals that immediate 
    action is absolutely essential for long-term survival and growth.
    """
    story.append(Paragraph(persuasive_text, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Page 3 - Major Risks & Concerns (Fearful/Cautious)
    story.append(Paragraph("Critical Risk Assessment", heading_style))
    
    risk_text = """
    We are deeply concerned about the escalating cybersecurity threats that have emerged this quarter. 
    There is a very real and immediate danger of significant data breaches that could compromise 
    customer information and damage our reputation irreparably. The potential financial impact 
    could exceed $50 million if these vulnerabilities are not addressed immediately.
    
    Furthermore, we are extremely worried about the supply chain disruptions affecting our primary 
    manufacturing partners. The geopolitical situation has created unprecedented uncertainty, and 
    we must develop contingency plans immediately. The risk of production delays is substantial 
    and could severely impact our Q2 delivery commitments.
    """
    story.append(Paragraph(risk_text, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Page 4 - Operational Challenges (Angry/Frustrated)
    story.append(Paragraph("Operational Performance Issues", heading_style))
    
    angry_text = """
    The recent system failure was completely unacceptable and reflects serious negligence in our 
    IT infrastructure management. This catastrophic failure caused massive disruption to our 
    operations and resulted in significant financial losses. We cannot tolerate such incompetence 
    in critical systems.
    
    The customer service metrics are absolutely appalling this quarter. The 25% decline in 
    satisfaction scores is completely unacceptable and demonstrates a fundamental failure in 
    our service delivery model. Immediate disciplinary action must be taken against responsible 
    managers. This situation is outrageous and requires immediate rectification.
    """
    story.append(Paragraph(angry_text, normal_style))
    story.append(PageBreak())
    
    # Page 5 - Financial Performance (Neutral/Informative)
    story.append(Paragraph("Financial Performance Analysis", heading_style))
    
    neutral_text = """
    According to the financial data, revenue reached $125.4 million this quarter, representing 
    an 8.7% increase over the previous quarter. Operating expenses totaled $89.2 million, 
    resulting in a net profit margin of 15.2%. The data indicates consistent performance 
    across most business units.
    
    Research findings show that market share has remained stable at 22.4%. Customer acquisition 
    costs have increased by 3.2% while customer lifetime value has decreased by 1.8%. These 
    metrics suggest we need to optimize our marketing efficiency.
    """
    story.append(Paragraph(neutral_text, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Page 6 - Strategic Opportunities (Excited/Optimistic)
    story.append(Paragraph("Strategic Growth Opportunities", heading_style))
    
    excited_text = """
    We are absolutely ecstatic about the breakthrough technology developed by our R&D team! 
    This revolutionary innovation has the potential to completely transform our industry and 
    establish us as market leaders for the next decade. The opportunity is massive and we 
    must move aggressively to capitalize on it.
    
    The market analysis reveals an amazing opportunity in the Asian markets that could triple 
    our revenue within 18 months. This is an extraordinary chance for exponential growth. 
    We should immediately allocate $25 million to accelerate our expansion plans. The timing 
    is perfect and the potential returns are absolutely incredible!
    """
    story.append(Paragraph(excited_text, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Page 7 - Employee Concerns (Sad/Disappointed)
    story.append(Paragraph("Employee Satisfaction & Culture", heading_style))
    
    sad_text = """
    Unfortunately, we have observed a deeply concerning decline in employee morale this quarter. 
    The recent survey results are truly disappointing, with a 30% increase in staff turnover 
    and a significant drop in engagement scores. This troubling trend reflects underlying 
    issues in our workplace culture that we must address immediately.
    
    It is with great regret that we report the departure of several key team members who 
    expressed dissatisfaction with our management approach. We have failed to provide the 
    supportive environment our employees deserve, and this failure has resulted in the loss 
    of valuable talent. We must do better.
    """
    story.append(Paragraph(sad_text, normal_style))
    story.append(PageBreak())
    
    # Page 8 - Compliance & Legal (Cautious/Fearful)
    story.append(Paragraph("Regulatory Compliance Update", heading_style))
    
    cautious_text = """
    We must carefully monitor the evolving regulatory landscape, as several new compliance 
    requirements could significantly impact our operations. The potential for regulatory 
    penalties is substantial if we fail to adapt our processes accordingly. We should 
    consider engaging external legal counsel to ensure we are fully compliant.
    
    There is a potential risk of litigation related to recent product performance issues. 
    While we believe our position is defensible, we must prepare for the possibility of 
    legal challenges. The financial implications could be material, and we need to 
    evaluate our insurance coverage carefully.
    """
    story.append(Paragraph(cautious_text, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Page 9 - Market Analysis (Mixed emotions)
    story.append(Paragraph("Market Position & Competitive Analysis", heading_style))
    
    mixed_text = """
    While we celebrate our strong market position, we must acknowledge the aggressive 
    moves by our main competitors. There is a real threat of market share erosion if 
    we don't respond effectively. However, we are optimistic about our new product 
    pipeline which should help maintain our leadership position.
    
    The economic uncertainty creates both challenges and opportunities. We need to 
    be cautious in our investments but should also consider strategic acquisitions 
    while valuations are favorable. This balanced approach will serve us well in 
    the current volatile environment.
    """
    story.append(Paragraph(mixed_text, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Page 10 - Conclusion & Call to Action (Strongly Persuasive)
    story.append(Paragraph("Conclusion & Strategic Imperatives", heading_style))
    
    conclusion_text = """
    In conclusion, while we face significant challenges, the overall outlook remains 
    strongly positive. We must act decisively and immediately on the recommendations 
    outlined in this report. Delaying action would be catastrophic for our competitive 
    position and long-term viability.
    
    We urgently need to: 1) Address the cybersecurity vulnerabilities immediately, 
    2) Invest aggressively in our digital transformation, 3) Restore employee morale 
    through meaningful cultural changes, and 4) Capitalize on the extraordinary 
    market opportunities we've identified.
    
    The board must approve these initiatives without hesitation. The future of our 
    organization depends on taking bold, immediate action. We cannot afford to wait!
    """
    story.append(Paragraph(conclusion_text, normal_style))
    
    # Build PDF
    doc.build(story)
    print(f"✅ Comprehensive test PDF created: {pdf_path}")
    print("\n📊 This PDF contains diverse emotional content perfect for testing:")
    print("   - Page 1: Optimistic/Excited with strong persuasion")
    print("   - Page 2: Mixed with very strong persuasive intent") 
    print("   - Page 3: Fearful/Cautious with major risk warnings")
    print("   - Page 4: Angry/Frustrated with operational issues")
    print("   - Page 5: Neutral/Informative financial data")
    print("   - Page 6: Excited/Optimistic about opportunities")
    print("   - Page 7: Sad/Disappointed about employee issues")
    print("   - Page 8: Cautious/Fearful about compliance risks")
    print("   - Page 9: Mixed emotions with balanced analysis")
    print("   - Page 10: Strongly persuasive conclusion with urgency")

def create_risk_focused_pdf():
    """Create another test PDF focused specifically on risks"""
    
    pdf_path = 'input_pdfs/risk_assessment_report.pdf'
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=1*inch)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle(
        name='Title',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        textColor=colors.HexColor('#DC143C')  # Red for danger
    )
    
    heading_style = ParagraphStyle(
        name='Heading2',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.HexColor('#34495E')
    )
    
    normal_style = styles['BodyText']
    warning_style = ParagraphStyle(
        name='Warning',
        parent=styles['BodyText'],
        backColor=colors.HexColor('#FFF0F0'),
        borderColor=colors.red,
        borderWidth=1,
        borderPadding=10
    )
    
    # Cover Page
    story.append(Paragraph("CRITICAL RISK ASSESSMENT REPORT", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("High-Priority Threats Requiring Immediate Attention", heading_style))
    story.append(Spacer(1, 0.5*inch))
    
    urgent_text = """
    <font color='red'><b>URGENT:</b></font> This report identifies several critical risks that require 
    immediate executive attention and decisive action. Failure to address these issues could 
    result in significant financial losses, reputational damage, and regulatory penalties.
    """
    story.append(Paragraph(urgent_text, warning_style))
    story.append(PageBreak())
    
    # Cybersecurity Risks
    story.append(Paragraph("1. Critical Cybersecurity Vulnerabilities", heading_style))
    cyber_risk = """
    <b>Risk Level: HIGH</b> - We have identified severe vulnerabilities in our network infrastructure 
    that could lead to catastrophic data breaches. Our penetration testing revealed that external 
    attackers could potentially access sensitive customer data and intellectual property.
    
    <b>Potential Impact:</b> $75+ million in direct costs, irreversible reputational damage, 
    regulatory fines up to $25 million, and potential class-action lawsuits.
    
    <b>Recommended Action:</b> Immediate security audit, patch deployment, and enhanced monitoring. 
    We must allocate $5 million for security upgrades in the next quarter.
    """
    story.append(Paragraph(cyber_risk, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Financial Risks
    story.append(Paragraph("2. Financial Compliance Risks", heading_style))
    financial_risk = """
    <b>Risk Level: HIGH</b> - Recent regulatory changes have created significant compliance gaps 
    in our financial reporting processes. We are currently not in full compliance with new 
    international accounting standards.
    
    <b>Potential Impact:</b> Regulatory penalties up to $15 million, stock price decline, 
    loss of investor confidence, and potential delisting from major exchanges.
    
    <b>Recommended Action:</b> Immediate engagement of external auditors, process overhaul, 
    and staff training. Compliance must be achieved within 90 days.
    """
    story.append(Paragraph(financial_risk, normal_style))
    story.append(PageBreak())
    
    # Operational Risks
    story.append(Paragraph("3. Supply Chain Disruption Risks", heading_style))
    supply_risk = """
    <b>Risk Level: MEDIUM-HIGH</b> - Geopolitical tensions and supplier concentration create 
    substantial risks to our production capabilities. A single supplier represents 45% of 
    our critical components.
    
    <b>Potential Impact:</b> Production delays up to 60 days, revenue loss of $40+ million, 
    customer contract penalties, and market share erosion.
    
    <b>Recommended Action:</b> Diversify supplier base, increase inventory buffers, and 
    develop alternative sourcing strategies within 6 months.
    """
    story.append(Paragraph(supply_risk, normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Talent Risks
    story.append(Paragraph("4. Critical Talent Retention Risks", heading_style))
    talent_risk = """
    <b>Risk Level: MEDIUM</b> - We are experiencing alarming turnover rates among key technical 
    staff. Competitors are actively poaching our top performers with aggressive compensation packages.
    
    <b>Potential Impact:</b> Loss of institutional knowledge, project delays, increased recruitment 
    costs, and potential intellectual property leakage.
    
    <b>Recommended Action:</b> Review compensation structure, enhance retention programs, 
    and accelerate knowledge transfer initiatives immediately.
    """
    story.append(Paragraph(talent_risk, normal_style))
    
    doc.build(story)
    print(f"✅ Risk-focused test PDF created: {pdf_path}")
    print("   - Focuses specifically on risk assessment with fearful/cautious tone")

if __name__ == "__main__":
    create_comprehensive_test_pdf()
    create_risk_focused_pdf()
    
    print("\n🎯 Both test PDFs are ready for SentimentScope analysis!")
    print("   They contain diverse emotional content, risks, and persuasive elements")
    print("   Perfect for testing the complete analytics system")