from langchain_community.document_loaders import PyPDFLoader
import os


# from langchain.document_loaders import PyPDFLoader
from pdf2image import convert_from_path
import pytesseract
from langchain.schema import Document


def read_pdf(pdf_path):
    '''returns a list of pages from the pdf file
    each page has two props: page_number and page_content
    '''
    real_pages = []

    # First attempt: Use PyPDFLoader
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()

        last_page_index = -1
        for page in pages:
            # I'm guessing these two prints are here for debugging purposes
            # print(f"Page {page.metadata['page']}: {page.page_content[:100]}...")    # 'metadata' holds info on the page, 'page_content[:100]' prints the first 100 items in the string contained in page_content
            # print('real_pages:', len(real_pages))
            page_index = page.metadata['page']
            if page_index == last_page_index:
                real_pages[-1].page_content += page.page_content    # not entirely sure, I think this checks for sequential pages indexed under the same number and if this is the case appends all of their info to the same element(page) instead of adding a new one
            else:
                real_pages.append(page)
                last_page_index = page_index

    except Exception as e:
        print(f'Error reading pdf {pdf_path} with PyPDFLoader: {e}')

    # If PyPDFLoader failed or returned empty results, try OCR
    if not real_pages:
        print('PyPDFLoader failed or returned no pages. Attempting to read PDF using OCR...')
        try:
            ocr_text = ocr_scanned_pdf(pdf_path)    # get OCR-retrieved text
            # Split the OCR text into pages
            ocr_pages = ocr_text.split('--- Page')      # use the defined page splits to separate the string where needed to define actual pages
            for i, page_content in enumerate(ocr_pages[1:], start=1):  # Skip the first split as it's empty (true as the first thing in the ocr_text string is '---Page')
                page_content = page_content.strip()     # remove unnecessary whitespace at front and end of string
                if page_content:
                    real_pages.append(Document(page_content=page_content, metadata={'page': i}))
        except Exception as ocr_error:
            print(f'Error reading pdf {pdf_path} with OCR: {ocr_error}')

    return real_pages

def ocr_scanned_pdf(pdf_path):
    text = ""
    images = convert_from_path(pdf_path)    # as usual, transforms pdf to image (as pytesseract works with images), special function used here since we only have a path for the pdf and its not read into memory already
    for i, image in enumerate(images):
        page_text = pytesseract.image_to_string(image)      # for each image, feed it to pytesseract and transform the read content into strings
        print(f"OCR result for page {i+1}: {page_text[:100]}...")
        text += f"--- Page {i+1} ---\n{page_text}\n\n"      # separate from previous pages and add to total text of OCR-scanned pdf
    return text


# if __name__ == '__main__':
#     # add current repo to path

#     pdf_path = '''./data/LEX_001.pdf'''
#     pages = read_pdf(pdf_path)

#     #concatenate all pages
#     text = ''
#     for page in pages:
#         text += page.page_content + '\n'

#     #save to file
#     with open('./data/LEX_001.txt', 'w') as f:
#         f.write(text)

#     print(pages)