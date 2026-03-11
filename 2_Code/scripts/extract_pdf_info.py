"""
Script to extract text information from bank credit assessment PDFs
This will help analyze the credit scoring criteria from different banks
"""

import os
import sys

def extract_pdf_text(pdf_path):
    """
    Extract text from PDF file using PyPDF2 or pdfplumber
    """
    try:
        # Try pdfplumber first (better for tables)
        import pdfplumber
        
        text_content = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    text_content.append(f"\n{'='*80}\nPage {page_num}\n{'='*80}\n")
                    text_content.append(text)
        
        return '\n'.join(text_content)
    
    except ImportError:
        print("pdfplumber not installed. Trying PyPDF2...")
        try:
            from PyPDF2 import PdfReader
            
            text_content = []
            reader = PdfReader(pdf_path)
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text:
                    text_content.append(f"\n{'='*80}\nPage {page_num}\n{'='*80}\n")
                    text_content.append(text)
            
            return '\n'.join(text_content)
        
        except ImportError:
            print("Error: Neither pdfplumber nor PyPDF2 is installed.")
            print("Please install one of them using:")
            print("  pip install pdfplumber")
            print("  or")
            print("  pip install PyPDF2")
            return None

def main():
    # Define PDF paths
    pdf_dir = r"d:\SystemFolders\Desktop\NCKH\docs\task\task1"
    output_dir = r"d:\SystemFolders\Desktop\NCKH\docs\task\task1\extracted"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # List of PDFs to process
    pdfs = [
        "ACB&HDB.pdf",
        "MB_VIETTINBANK.pdf",
        "Techcombank&VPBank.pdf",
        "VCB&BIDV.pdf"
    ]
    
    for pdf_name in pdfs:
        pdf_path = os.path.join(pdf_dir, pdf_name)
        
        if not os.path.exists(pdf_path):
            print(f"Warning: {pdf_name} not found at {pdf_path}")
            continue
        
        print(f"\nProcessing {pdf_name}...")
        
        # Extract text
        text = extract_pdf_text(pdf_path)
        
        if text:
            # Save to text file
            output_name = pdf_name.replace('.pdf', '.txt')
            output_path = os.path.join(output_dir, output_name)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"  ✓ Extracted to {output_path}")
            print(f"  ✓ Character count: {len(text)}")
        else:
            print(f"  ✗ Failed to extract text from {pdf_name}")

if __name__ == "__main__":
    main()
