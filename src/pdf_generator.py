from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from pathlib import Path
import logging
from typing import Dict, Optional, List
from datetime import datetime
import matplotlib.pyplot as plt
import io

logger = logging.getLogger('PDFGenerator')

class PDFGenerator:
    """Handles generation of PDF documents for video analysis results"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger('PDFGenerator')
        self.styles = getSampleStyleSheet()
        
        # Create custom styles
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        ))
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12
        ))
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=8
        ))

    def generate_summary_pdf(self, summary_info: Dict) -> Optional[Path]:
        """Generate PDF document for video summary"""
        try:
            if 'video_info' not in summary_info:
                self.logger.error("Missing video_info in summary_info")
                return None
                
            if 'video_id' not in summary_info['video_info']:
                self.logger.error("Missing video_id in video_info")
                return None
                
            video_id = summary_info['video_info']['video_id']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.output_dir / f"summary_{video_id}_{timestamp}.pdf"
            
            doc = SimpleDocTemplate(
                str(output_file),
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Build content
            story = []
            
            # Title
            title = Paragraph(f"Video Summary: {summary_info['video_info']['title']}", 
                            self.styles['CustomTitle'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Metadata
            metadata = [
                ['Video ID:', video_id],
                ['Duration:', f"{summary_info['video_info']['duration']} seconds"],
                ['Processing Time:', f"{summary_info['processing_time']:.2f} seconds"],
                ['Compression Ratio:', f"{summary_info['processing_stats']['compression_ratio']:.2%}"]
            ]
            
            meta_table = Table(metadata, colWidths=[100, 300])
            meta_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey)
            ]))
            story.append(meta_table)
            story.append(Spacer(1, 20))
            
            # Summary
            story.append(Paragraph("Summary", self.styles['CustomHeading']))
            story.append(Paragraph(summary_info['summary'], self.styles['CustomBody']))
            story.append(Spacer(1, 20))
            
            # Content Analysis
            if 'content_analysis' in summary_info:
                story.append(Paragraph("Content Analysis", self.styles['CustomHeading']))
                
                # Content Types
                if 'content_types' in summary_info['content_analysis']:
                    content_types = summary_info['content_analysis']['content_types']
                    content_data = [[type_, f"{percentage:.1f}%"] 
                                for type_, percentage in content_types.items()]
                    
                    content_table = Table([['Content Type', 'Percentage']] + content_data,
                                        colWidths=[200, 100])
                    content_table.setStyle(TableStyle([
                        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 0), (-1, -1), 10),
                        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                        ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey)
                    ]))
                    story.append(content_table)
                    story.append(Spacer(1, 12))
                
                # Key Concepts
                if 'key_concepts' in summary_info['content_analysis']:
                    story.append(Paragraph("Key Concepts", self.styles['CustomHeading']))
                    concepts = ", ".join(summary_info['content_analysis']['key_concepts'])
                    story.append(Paragraph(concepts, self.styles['CustomBody']))
            
            # Build PDF
            doc.build(story)
            self.logger.info(f"Summary PDF generated: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error generating summary PDF: {str(e)}", exc_info=True)
            return None

    def generate_transcript_pdf(self, transcript: str, video_info: Dict) -> Optional[Path]:
        """Generate PDF document for video transcript"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.output_dir / f"transcript_{video_info['video_id']}_{timestamp}.pdf"
            
            doc = SimpleDocTemplate(
                str(output_file),
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            story = []
            
            # Title
            title = Paragraph(f"Video Transcript: {video_info['title']}", 
                            self.styles['CustomTitle'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Metadata
            metadata = [
                ['Video ID:', video_info['video_id']],
                ['Duration:', f"{video_info['duration']} seconds"],
                ['Transcript Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            ]
            
            meta_table = Table(metadata, colWidths=[100, 300])
            meta_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey)
            ]))
            story.append(meta_table)
            story.append(Spacer(1, 20))
            
            # Transcript text
            story.append(Paragraph("Full Transcript", self.styles['CustomHeading']))
            
            # Split transcript into paragraphs for better formatting
            paragraphs = transcript.split('\n\n')
            for para in paragraphs:
                story.append(Paragraph(para, self.styles['CustomBody']))
                story.append(Spacer(1, 12))
            
            # Build PDF
            doc.build(story)
            self.logger.info(f"Transcript PDF generated: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error generating transcript PDF: {str(e)}")
            return None

    def generate_visualization_report(self, summary_info: Dict) -> Optional[Path]:
        """Generate PDF report for visualizations with analysis"""
        try:
            if 'video_info' not in summary_info:
                self.logger.error("Missing video_info in summary_info")
                return None
                
            if 'video_id' not in summary_info['video_info']:
                self.logger.error("Missing video_id in video_info")
                return None
                
            video_id = summary_info['video_info']['video_id']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.output_dir / f"visualization_report_{video_id}_{timestamp}.pdf"
            
            doc = SimpleDocTemplate(
                str(output_file),
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            story = []
            
            # Title
            title = Paragraph(f"Video Analysis Report: {summary_info['video_info']['title']}", 
                            self.styles['CustomTitle'])
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Content Type Distribution
            if 'content_analysis' in summary_info:
                story.append(Paragraph("Content Distribution Analysis", self.styles['CustomHeading']))
                content_types = summary_info['content_analysis'].get('content_types', {})
                if content_types:
                    # Create pie chart
                    plt.figure(figsize=(8, 6))
                    plt.pie(content_types.values(), labels=content_types.keys(), autopct='%1.1f%%')
                    plt.title("Content Type Distribution")
                    
                    # Save to bytes buffer
                    img_data = io.BytesIO()
                    plt.savefig(img_data, format='png', bbox_inches='tight')
                    img_data.seek(0)
                    plt.close()
                    
                    # Add to PDF
                    img = Image(img_data, width=400, height=300)
                    story.append(img)
                    story.append(Spacer(1, 12))
                    
                    # Add analysis text
                    analysis = "The content analysis shows the following distribution:\n\n"
                    for content_type, percentage in content_types.items():
                        analysis += f"• {content_type}: {percentage:.1f}%\n"
                    story.append(Paragraph(analysis, self.styles['CustomBody']))
                    story.append(Spacer(1, 20))
            
            # Sentiment Analysis
            if 'sentiment_analysis' in summary_info:
                story.append(Paragraph("Sentiment Analysis", self.styles['CustomHeading']))
                sentiment_data = summary_info['sentiment_analysis']
                
                if 'average_score' in sentiment_data:
                    avg_score = sentiment_data['average_score']
                    sentiment_text = f"Average Sentiment Score: {avg_score:.2f}\n\n"
                    
                    if avg_score > 0.6:
                        sentiment_text += "The content demonstrates predominantly positive sentiment."
                    elif avg_score < 0.4:
                        sentiment_text += "The content demonstrates predominantly negative sentiment."
                    else:
                        sentiment_text += "The content demonstrates neutral sentiment."
                        
                    story.append(Paragraph(sentiment_text, self.styles['CustomBody']))
                    story.append(Spacer(1, 20))
            
            # Processing Statistics
            if 'processing_stats' in summary_info:
                story.append(Paragraph("Processing Statistics", self.styles['CustomHeading']))
                stats = summary_info['processing_stats']
                
                stats_data = [
                    ['Metric', 'Value'],
                    ['Processing Time', f"{stats['processing_time']:.2f} seconds"],
                    ['Compression Ratio', f"{stats['compression_ratio']:.2%}"],
                    ['Original Length', f"{stats['original_length']} words"],
                    ['Summary Length', f"{stats['summary_length']} words"]
                ]
                
                stats_table = Table(stats_data, colWidths=[200, 200])
                stats_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey)
                ]))
                story.append(stats_table)
            
            # Build PDF
            doc.build(story)
            self.logger.info(f"Visualization report generated: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error generating visualization report: {str(e)}", exc_info=True)
            return None