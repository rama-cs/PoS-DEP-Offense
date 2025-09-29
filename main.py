import spacy
import sentencepiece as spm
from datasets import load_dataset
import re

# --- 1. SETUP PREPROCESSING TOOLS ---
# As described in Section 4.3 and 4.4 of the paper.

print("Loading preprocessing tools...")
# Load spaCy for English POS and Dependency parsing 
try:
    nlp_en = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy English model not found. Please run: python -m spacy download en_core_web_sm")
    exit()
    
# NOTE: The paper mentions "ThamizhiUD" for Tamil parsing.
# As this may not be a standard package, we'll use a placeholder function
# to represent its role in a real-world implementation.
def parse_tamil_syntax(text):
    """Placeholder for a Tamil UD-compliant parser like ThamizhiUD."""
    # In a real scenario, this function would return word, POS tag, and DEP head info.
    # For this example, we'll return dummy tags.
    words = text.split()
    return [(word, "NOUN", 0) for word in words] # (token, pos_tag, head_index)

# Load a SentencePiece model. The paper trained one with 32K merge operations.
# You would replace 'spm.model' with the path to your trained model.
# For this example, let's assume a model file exists.
# sp = spm.SentencePieceProcessor(model_file='path/to/your/spm.model')

print("Tools loaded successfully.")

# --- 2. DEFINE THE PREPROCESSING PIPELINE ---

def normalize_text(text):
    """
    Performs text normalization as described in Section 4.4[cite: 151].
    - Removes extraneous characters
    - Standardizes punctuation
    """
    text = text.lower()
    text = re.sub(r'[\n\t]', ' ', text) # Remove newlines and tabs
    text = re.sub(r'[^\w\s\d.,?!]', '', text) # Keep essential punctuation and word chars
    text = re.sub(r'\s+', ' ', text).strip() # Standardize whitespace
    return text

def align_tags_to_subwords(words_with_tags, subword_tokens):
    """
    Aligns word-level POS/DEP tags to subword tokens.
    Strategy: The first subword of a word gets the original tag, subsequent
    subwords get a special "continuation" tag or simply inherit the tag.
    """
    word_queue = list(words_with_tags)
    aligned_pos_tags = []
    aligned_dep_heads = []
    current_word, current_pos, current_dep = "", "PAD", -1
    
    if word_queue:
        current_word, current_pos, current_dep = word_queue.pop(0)

    for token in subword_tokens:
        aligned_pos_tags.append(current_pos)
        aligned_dep_heads.append(current_dep)
        
        # SentencePiece uses a prefix ' ' to mark the start of a new word
        if token.startswith(' ') and token.lstrip(' ') in current_word:
             # This token is the start of the next word, so pop from queue
             if word_queue:
                 current_word, current_pos, current_dep = word_queue.pop(0)
    
    return aligned_pos_tags, aligned_dep_heads


def preprocess_function(examples):
    """
    The main function to apply all preprocessing steps to a batch of examples.
    """
    # For simplicity, we'll use dummy tokenization here. 
    # Replace with your actual SentencePiece processor.
    # e.g., inputs = sp.encode(batch_of_english_sentences, out_type=str)
    
    # --- Process English Source ---
    en_texts = [normalize_text(text) for text in examples["en"]]
    
    # Get syntactic annotations [cite: 150]
    en_docs = nlp_en.pipe(en_texts)
    en_words_with_tags = [
        [(token.text, token.pos_, token.head.i) for token in doc] for doc in en_docs
    ]
    
    # This is a placeholder for actual SentencePiece tokenization
    en_tokens_subword = [text.split() for text in en_texts]

    # --- Process Tamil Target ---
    ta_texts = [normalize_text(text) for text in examples["ta"]]
    
    # Get syntactic annotations (using our placeholder)
    ta_words_with_tags = [parse_tamil_syntax(text) for text in ta_texts]

    # This is a placeholder for actual SentencePiece tokenization
    ta_tokens_subword = [text.split() for text in ta_texts]
    
    # --- Placeholder for Alignment and ID Conversion ---
    # In a real implementation, you would:
    # 1. Align tags to subwords using a function like `align_tags_to_subwords`.
    # 2. Convert all tokens (en, ta, pos, dep) to integer IDs using vocabularies.
    # 3. Add offensive language labels from your annotated corpus[cite: 145].

    # For this example, we'll just return the tokenized text.
    model_inputs = {
        "english_text": en_texts,
        "tamil_text": ta_texts,
        "english_tokens": en_tokens_subword,
        "tamil_tokens": ta_tokens_subword,
        "english_syntax": en_words_with_tags
    }
    return model_inputs

# --- 3. LOAD AND PROCESS THE DATASET ---

print("\nLoading dataset...")
# Load a dataset. Samanantar is one of the corpora mentioned in the paper.
# We'll take a small subset for a quick demonstration.
raw_dataset = load_dataset("ai4bharat/samanantar", "en-ta", split='train', streaming=True).take(5)

# Convert the iterable dataset to a standard dictionary format for mapping
raw_dict = {"en": [], "ta": []}
for item in raw_dataset:
    raw_dict["en"].append(item["translation"]["en"])
    raw_dict["ta"].append(item["translation"]["ta"])
    
from datasets import Dataset
dataset = Dataset.from_dict(raw_dict)

print("Dataset loaded. Starting preprocessing...")
# Apply the preprocessing function to the entire dataset
# The `map` function processes the data in batches for efficiency.
processed_dataset = dataset.map(preprocess_function, batched=True)

print("Preprocessing complete.")

# --- 4. INSPECT THE PROCESSED DATA ---
print("\n--- Example of a Processed Entry ---")
example = processed_dataset[0]
print(f"Original English: {example['english_text']}")
print(f"Original Tamil:   {example['tamil_text']}")
print("\n--- English Syntactic Annotations (Token, POS, Head Index) ---")
print(example["english_syntax"])
print("--------------------------------------")