<h1 align="center">
    OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation
</h1>

[Dataset (🤗Hugging Face)](https://huggingface.co/datasets/opendatalab/OHR-Bench) | [Dataset (OpenDataLab)]()

This repository contains the official code of **OHR-Bench**, a benchmark designed to evaluate the cascading impact of OCR on RAG.

# Overview
- **PDF, gt structured data and Q&A datasets: [[🤗 Hugging Face](https://huggingface.co/datasets/opendatalab/OHR-Bench)] `data/pdfs`, `data/ground_truth_structured_data` and `data/qas`**. It includes 4000+ unstructured PDF pages from various domains, including Textbook, Law, Finance, Newspaper, Manual and Academia and Q&A datasets sourced from multimodal document elements. Each PDF page is equipped with a human-verified ground truth structured data.
- **Perturbed data with OCR errors: [[🤗 Hugging Face](https://huggingface.co/datasets/opendatalab/OHR-Bench)] `data/perturbed_structured_data`**. In order to conduct in-depth analysis of the OCR's impact on RAG, OHR-Bench identifies *Semantic Noise* and *Formatting Noise* and introduce them with mild, moderate and severe perturbation based on real-world OCR errors.
- **Evaluation framework: [[Github opendatalab/OHR-Bench](https://github.com/opendatalab/OHR-Bench)]**. We provide a RAG evaluation framework to assess the impact of OCR processed structured data and our perturbed data on RAG including retrieval, generation and overall performance.

![framework](./figs/framework.png)

## Evaluation Results
![img.png](./figs/results.png)

We evaluate the suitability of current OCR solutions for real-world RAG applications by conducting comprehensive experiments with our OHR-Bench.
We derive conclusions as follows:

- Pipeline-based OCR demonstrates the best performance. Employing Marker achieves the best retrieval performance across all OCR solutions, while MinerU dominates the generation and overall evaluation.
- All OCR solutions suffer performance degradation. Even the best solutions show a decrease of 1.9 in EM@1 and 2.93 F1@1 in the overall evaluation, with greater losses in the retrieval and generation stages.

# Getting Started
## Installation
```bash
pip install -r requirements.txt
```

## Dataset preparation
### Generation and end-to-end evaluation
Place the Q&A JSON files in the data/qa directory. Each JSON file should be structured as follows:

<details>
<summary>Q&A JSON</summary>

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

</details>

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

# Overall Results

# Acknowledgement
The evaluation framework is based on [CRUD](https://github.com/IAAR-Shanghai/CRUD_RAG), thanks so much for this brilliant project.

# Citation
```
@article{zhang2024ocr,
  title={OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation},
  author={Junyuan Zhang and Qintong Zhang and Bin Wang and Linke Ouyang and Zichen Wen and Ying Li and Ka-Ho Chow and Conghui He and Wentao Zhang},
  journal={arXiv preprint arXiv:2412.02592},
  year={2024}
}
```

# Copyright Statement
The PDFs are collected from public online channels and community user contributions. Content that is not allowed for distribution has been removed. The dataset is for research purposes only and not for commercial use. If there are any copyright concerns, please contact OpenDataLab@pjlab.org.cn.
