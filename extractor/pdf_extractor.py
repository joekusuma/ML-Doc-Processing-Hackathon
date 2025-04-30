import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_bytes
import numpy as np
import os
import json
from pathlib import Path
import markdown
import re
import camelot

class PDFExtractor:
    """
    Class for extracting text from PDF documents using PyMuPDF and OCR if needed
    """
    
    def __init__(self, use_ocr=False, ocr_lang='eng'):
        """
        Initialize the PDF extractor
        
        Args:
            use_ocr (bool): Whether to use OCR for text extraction
            ocr_lang (str): Language for OCR
        """
        self.use_ocr = use_ocr
        self.ocr_lang = ocr_lang
        
        # Configure pytesseract path if needed
        # pytesseract.pytesseract.tesseract_cmd = r'path_to_tesseract_executable'
    
    def extract_text_pdf(self, pdf_file):
        """
        Extract text from PDF using PyMuPDF
        
        Args:
            pdf_file: File object or path to PDF
            
        Returns:
            dict: Extracted text with page numbers as keys
        """
        try:
            # Check if input is bytes or file path
            if isinstance(pdf_file, bytes):
                doc = fitz.open(stream=pdf_file, filetype="pdf")
            else:
                doc = fitz.open(pdf_file)
            
            extracted_text = {}
            
            for page_num, page in enumerate(doc):
                if self.use_ocr:
                    # Convert page to image and use OCR
                    pix = page.get_pixmap()
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, pix.n
                    )
                    text = pytesseract.image_to_string(img, lang=self.ocr_lang)
                else:
                    # Use PyMuPDF's text extraction
                    text = page.get_text()
                
                extracted_text[page_num] = text.strip()
            
            doc.close()
            return extracted_text
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def extract_from_bytes(self, pdf_bytes):
        """
        Extract text from PDF bytes
        
        Args:
            pdf_bytes (bytes): PDF content as bytes
            
        Returns:
            dict: Extracted text with page numbers as keys
        """
        try:
            return self.extract_text_pdf(pdf_bytes)
        except Exception as e:
            raise Exception(f"Error extracting text from PDF bytes: {str(e)}")
    
    def extract_text_with_ocr(self, pdf_bytes):
        """
        Extract text using OCR regardless of use_ocr setting
        
        Args:
            pdf_bytes (bytes): PDF content as bytes
            
        Returns:
            dict: Extracted text with page numbers as keys
        """
        try:
            # Convert PDF to images
            images = convert_from_bytes(pdf_bytes)
            
            extracted_text = {}
            
            for i, image in enumerate(images):
                # Perform OCR on each page
                text = pytesseract.image_to_string(image, lang=self.ocr_lang)
                extracted_text[i] = text.strip()
            
            return extracted_text
        except Exception as e:
            raise Exception(f"Error extracting text with OCR: {str(e)}")
    
    def extract_text_from_markdown(self, markdown_content):
        """
        Extract plain text from markdown content
        
        Args:
            markdown_content (str): Markdown content
            
        Returns:
            str: Plain text extracted from markdown
        """
        try:
            # Convert markdown to HTML
            html = markdown.markdown(markdown_content)
            
            # Remove HTML tags to get plain text
            text = re.sub(r'<[^>]+>', '', html)
            
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from markdown: {str(e)}")


    def extract_tables_from_pdf(self, pdf_file):
        """
        Extract tables from PDF using multiple methods for better results
        
        Args:
            pdf_file: File object or path to PDF
            
        Returns:
            list: List of extracted tables as lists of lists
        """
        try:
            # First attempt: Use Camelot for table detection
            tables = []
            
            # If input is bytes, save to a temporary file
            if isinstance(pdf_file, bytes):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(pdf_file)
                    tmp_path = tmp.name
                
                # Extract tables with Camelot
                camelot_tables = camelot.read_pdf(tmp_path, pages='all', flavor='lattice')
                if len(camelot_tables) == 0:
                    camelot_tables = camelot.read_pdf(tmp_path, pages='all', flavor='stream')
                
                # Clean up temporary file
                import os
                os.unlink(tmp_path)
            else:
                # Extract tables with Camelot (file path)
                camelot_tables = camelot.read_pdf(pdf_file, pages='all', flavor='lattice')
                if len(camelot_tables) == 0:
                    camelot_tables = camelot.read_pdf(pdf_file, pages='all', flavor='stream')
            
            # Convert Camelot tables to list format
            for table in camelot_tables:
                tables.append(table.df.values.tolist())
            
            # If Camelot found tables, return them
            if tables:
                return tables
            
            # Fallback to PyMuPDF's table detection
            return self._extract_tables_pymupdf(pdf_file)
            
        except Exception as e:
            print(f"Error using Camelot: {str(e)}")
            return self._extract_tables_pymupdf(pdf_file)
    
    def _extract_tables_pymupdf(self, pdf_file):
        """Original PyMuPDF table extraction method as fallback"""
        try:
            # Check if input is bytes or file path
            if isinstance(pdf_file, bytes):
                doc = fitz.open(stream=pdf_file, filetype="pdf")
            else:
                doc = fitz.open(pdf_file)
            
            tables = []
            
            for page_num, page in enumerate(doc):
                # Extract tables using PyMuPDF's built-in table detection
                tab = page.find_tables()
                if tab.tables:
                    for t in tab.tables:
                        rows = []
                        for cells in t.cells:
                            row = []
                            for cell in cells:
                                rect = fitz.Rect(cell.bbox)
                                text = page.get_text("text", clip=rect)
                                row.append(text.strip())
                            rows.append(row)
                        tables.append(rows)
            
            doc.close()
            return tables
            
        except Exception as e:
            raise Exception(f"Error extracting tables from PDF: {str(e)}")
    
    def compare_extraction_methods(self, pdf_bytes):
        """
        Compare extraction results between PyMuPDF and OCR
        
        Args:
            pdf_bytes (bytes): PDF content as bytes
            
        Returns:
            dict: Comparison results
        """
        # Save current OCR setting
        current_ocr_setting = self.use_ocr
        
        # Extract with PyMuPDF
        self.use_ocr = False
        pymupdf_result = self.extract_from_bytes(pdf_bytes)
        
        # Extract with OCR
        ocr_result = self.extract_text_with_ocr(pdf_bytes)
        
        # Restore original setting
        self.use_ocr = current_ocr_setting
        
        # Calculate text length differences to compare extraction quality
        comparison = {}
        for page_num in pymupdf_result:
            if page_num in ocr_result:
                pymupdf_len = len(pymupdf_result[page_num])
                ocr_len = len(ocr_result[page_num])
                
                comparison[page_num] = {
                    "pymupdf_length": pymupdf_len,
                    "ocr_length": ocr_len,
                    "difference_percentage": abs(pymupdf_len - ocr_len) / max(pymupdf_len, ocr_len) * 100 if max(pymupdf_len, ocr_len) > 0 else 0
                }
        
        return comparison