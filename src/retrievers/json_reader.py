"""JSON Reader."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.readers.base import BaseReader
from llama_index.schema import Document

class PCJSONReader(BaseReader):
    """JSON reader.

    Reads JSON documents with Unstructured
    """
    def __init__(self) -> None:
        """Initialize with arguments."""
        super().__init__()

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Load data from the input file."""
        documents = []
        with open(file, encoding="utf-8") as f:
            load_data = json.load(f)
            for data in load_data:
                documents.append(Document(text=data["text"], metadata={
                    "page_idx": data["page_idx"] if "page_idx" in data else data["page_no"],
                    **extra_info
                }))
        return documents