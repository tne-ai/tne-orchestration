import io

from v2.etl.json_util import write_json
from v2.etl.typer_util import TyperEnum, typer_arg, typer_opt


def _pdfminer_extract_pdf_text(pdf_fp):
    # Lazy import only if this extracter is used.
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
    from pdfminer.pdfpage import PDFPage

    pdf_resource_manager = PDFResourceManager()
    string_io = io.StringIO()
    la_params = LAParams()
    text_converter = TextConverter(pdf_resource_manager, string_io, laparams=la_params)
    pdf_page_interpreter = PDFPageInterpreter(pdf_resource_manager, text_converter)
    magic_page_separator = "<MAGIC_PAGE_SEPARATOR>"
    for page in PDFPage.get_pages(pdf_fp):
        pdf_page_interpreter.process_page(page)
        string_io.write(magic_page_separator)
    page_texts = string_io.getvalue().split(magic_page_separator)
    assert len(page_texts[-1]) == 0
    return page_texts[:-1]


def _pdfplumber_extract_pdf_text(pdf_fp):
    # Lazy import only if this extracter is used.
    import pdfplumber

    pdf_doc = pdfplumber.open(pdf_fp)
    page_texts = [page.extract_text() for page in pdf_doc.pages]
    return page_texts


def _pymupdf_extract_pdf_text(pdf_fp):
    # Lazy import only if this extracter is used.
    import fitz

    pdf_doc = fitz.open(pdf_fp)
    page_texts = [page.get_text() for page in pdf_doc]
    return page_texts


def _pypdf_extract_pdf_text(pdf_fp):
    # Lazy import only if this extracter is used.
    import pypdf

    pdf_doc = pypdf.PdfReader(pdf_fp)
    page_texts = [page.extract_text() for page in pdf_doc.pages]
    return page_texts


def _pypdfium2_extract_pdf_text(pdf_fp):
    # Lazy import only if this extracter is used.
    import pypdfium2

    page_texts = []
    pdf_doc = pypdfium2.PdfDocument(pdf_fp)
    try:
        for page in pdf_doc:
            try:
                text_page = page.get_textpage()
                try:
                    page_text = text_page.get_text_range()
                    page_texts.append(page_text)
                finally:
                    text_page.close()
            finally:
                page.close()
    finally:
        pdf_doc.close()
    return page_texts


class _PdfExtracterId(TyperEnum):
    pdfminer = "pdfminer", _pdfminer_extract_pdf_text
    pdfplumber = "pdfplumber", _pdfplumber_extract_pdf_text
    pymupdf = "pymupdf", _pymupdf_extract_pdf_text
    pypdf = "pypdf", _pypdf_extract_pdf_text
    pypdfium2 = "pypdfium2", _pypdfium2_extract_pdf_text


def extract_pdf_text(
    input_pdf_file: typer_arg(str, "TODO: Add help text."),
    output_json_file: typer_arg(str, "TODO: Add help text."),
    pdf_extracter_id: typer_opt(
        _PdfExtracterId, "TODO: Add help text."
    ) = _PdfExtracterId.pypdfium2,
):
    """TODO: Add help text."""

    extract_pdf_text_fun = pdf_extracter_id.extra_value

    with open(input_pdf_file, "rb") as pdf_fp:
        page_texts = extract_pdf_text_fun(pdf_fp)

    doc_obj = {
        "command_args": {
            "extract_pdf_text_args": {
                "input_pdf_file": input_pdf_file,
                "output_json_file": output_json_file,
                "pdf_extracter_id": pdf_extracter_id.name,
            },
        },
        "pages": [
            {
                "raw_extracted_text": page_text,
            }
            for page_text in page_texts
        ],
    }

    write_json(output_json_file, doc_obj)
