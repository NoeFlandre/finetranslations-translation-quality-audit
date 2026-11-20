"""Where the source data, the index and the results live on the Hugging Face Hub."""

DATASET = "HuggingFaceFW/finetranslations-edu"
DATA_GLOB = f"datasets/{DATASET}/data/*.parquet"
DEFAULT_REPO = "NoeFlandre/finetranslations-translation-quality-audit"
INDEX_PATH = "index/language_groups.parquet"
INDEX_LOCATION = f"datasets/{DEFAULT_REPO}/{INDEX_PATH}"  # the index describes the source data
RESULTS_PATH = "v1/audit.parquet"
INDEXED_LANGUAGES = ("fra_Latn", "deu_Latn", "rus_Cyrl", "cmn_Hani")  # the index covers these only
