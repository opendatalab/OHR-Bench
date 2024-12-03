<h1 align="center">
    OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation
</h1>

This repository contains the official code of **OHR-Bench**, a benchmark designed to evaluate the cascading impact of OCR on RAG.

# Overview
- **PDF, gt structured data and Q&A datasets: [[🤗 Hugging Face](https://huggingface.co/datasets/opendatalab/OHR-Bench)] `data/pdfs`, `data/ground_truth_structured_data` and `data/qas`**. It includes 4000+ unstructured PDF pages from various domains, including Textbook, Law, Finance, Newspaper, Manual and Academia and Q&A datasets sourced from multimodal document elements. Each PDF page is equipped with a human-verified ground truth structured data.
- **Perturbed data with OCR errors: [[🤗 Hugging Face](https://huggingface.co/datasets/opendatalab/OHR-Bench)] `data/perturbed_structured_data`**. In order to conduct in-depth analysis of the OCR's impact on RAG, OHR-Bench identifies *Semantic Noise* and *Formatting Noise* and introduce them with mild, moderate and severe perturbation based on real-world OCR errors.
- **Evaluation framework: [[Github opendatalab/OHR-Bench](https://github.com/opendatalab/OHR-Bench)]**. We provide a RAG evaluation framework to assess the impact of OCR processed structured data and our perturbed data on RAG including retrieval, generation and overall performance.

![framework](./figs/framework.png)

# Getting Started
## Installation
```bash
pip install -r requirements.txt
```

## Dataset preparation
### Generation and end-to-end evaluation
Place the Q&A JSON files in the data/qa directory. Each JSON file should be structured as follows:
```json
[
    {
        "doc_name": "finance/JPMORGAN_2021Q1_10Q", // Document source
        "ID": "00073cc2-c801-467c-9039-fca63c78c6a9", // Unique ID
        "questions": "What was the total amount of nonaccrual loans retained as of March 31, 2021?",
        "answers": "842",
        // Th relevant context used to answer the question. In generation evaluation, it is the parsed results of the PDF page that questions derived from. In end-to-end evaluation, it is the retrieved results.
        "context": "Selected metrics\n...",
        "doc_type": "finance", // PDF domain.
        "difficulty_level": "Easy",
        "answer_form": "Numeric", // Answer format.
        "evidence_source": "table", // Evidence source.
        "evidence_context": "Nonaccrual loans retained $^{(\\mathrm{a})}$ & \\$ & 842 & \\$ & 689 & $22 \\%$", // Evidence.
        "evidence_page_no": 24
    },
    ...
]
```
Refer to the example in `data/qa/gt` for more details.
### Retrieval
Place the parsed structured data in the `data/retrieval_base` directory. Refer to the example in `data/retrieval_base/gt` for more details.
```bash
# Directory structure
retrieval_base/gt/ # gt or OCR parsed results
├── finance # domain
│   ├── 3M_2023Q2_10Q.json # parsed results
│   ├── ...
├── textbook
...
```

# Run Evaluation
```bash
# Generation
bash shell/generation.sh gt finance qwen2_7b
# Retrieval
bash shell/retrieval.sh gt finance qwen2_7b
# End-to-end
# Run the corresponding retrieval before end-to-end evaluation
bash shell/end2end.sh gt finance qwen2_7b
```

# Acknowledgement
The evaluation framework is based on [CRUD](https://github.com/IAAR-Shanghai/CRUD_RAG), thanks so much for this brilliant project.

# CITATION
```
```
