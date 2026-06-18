# Document Parsing

This repo groups the document extraction work under `document_parsing/`.

## Main Parts

- `document_parsing/casestudy/`  
  Case study extraction code, prompts, schema, evaluation scripts, and API runners.

- `document_parsing/birthcert/`  
  Birth certificate extraction code, prompts, schema, OCR/YOLO helpers, and evaluation scripts.

- `document_parsing/data/`, `document_parsing/outputs/`, `document_parsing/tools/`  
  Input images, labels, generated results, model artifacts, notebooks, examples, and helper scripts used by the parsers.

Example commands:

```powershell
python -m document_parsing.casestudy.gemini_infer
python -m document_parsing.birthcert.gemini_infer
python -m document_parsing.casestudy.parser.validate --input-dir document_parsing/outputs/predictions
```
