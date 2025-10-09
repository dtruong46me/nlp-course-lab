"""
Refactored script for training and evaluating a spaCy NER model on resume data.

This script handles:
1. Converting data from Dataturks JSON format to spaCy's format.
2. Training a new NER model from scratch.
3. Evaluating the trained model's performance.
4. Saving the model's predictions on test data to text files.

To run this script, you need to install spaCy and its English models:
pip install spacy
python -m spacy download en_core_web_sm
"""

import json
import logging
import random
import pathlib
from typing import List, Tuple, Dict, Any, Optional

# Third-party imports
import spacy
from spacy.tokens import Doc, DocBin
from spacy.training.example import Example
from sklearn.model_selection import train_test_split

# --- Constants ---
# Using pathlib for robust path management
BASE_DIR = pathlib.Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "traindata.json"
TEST_DATA_PATH = BASE_DIR / "data" / "testdata.json"
OUTPUT_DIR = BASE_DIR / "output_model"
PREDICTIONS_DIR = BASE_DIR / "resume_predictions"
N_ITERATIONS = 10
DROPOUT_RATE = 0.2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Type Aliases for clarity
Entity = Tuple[int, int, str]
SpacyData = Tuple[str, Dict[str, List[Entity]]]

# Đảm bảo bạn đã import thư viện này ở đầu file
from spacy.util import filter_spans

# Import thêm thư viện này ở đầu file
from spacy.util import filter_spans
import string # Thư viện chuẩn của Python

def convert_dataturks_to_spacy(file_path: pathlib.Path) -> Optional[List[SpacyData]]:
    """
    Converts data from Dataturks JSON-lines format to spaCy's training format.
    
    This function automatically handles three common data issues:
    1. Trims leading/trailing whitespace and punctuation from entities (fixes spaCy error E024).
    2. Filters out misaligned entities (fixes spaCy warning W030).
    3. Filters out overlapping entities, keeping the longest one (fixes spaCy error E103).
    """
    try:
        training_data: List[SpacyData] = []
        # Định nghĩa các ký tự cần cắt tỉa
        PUNCTUATION_TO_TRIM = string.punctuation + string.whitespace
        
        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                text = data['content']
                entities = []
                
                if data['annotation'] is None:
                    continue
                    
                for annotation in data['annotation']:
                    point = annotation['points'][0]
                    labels = annotation['label']
                    if not isinstance(labels, list):
                        labels = [labels]

                    for label in labels:
                        start, end = point['start'], point['end'] + 1
                        
                        # --- PHẦN FIX LỖI MỚI: Tự động "cắt tỉa" (trim) thực thể ---
                        entity_text = text[start:end]
                        
                        # 1. Cắt bỏ khoảng trắng/dấu câu ở hai đầu
                        trimmed_text = entity_text.strip(PUNCTUATION_TO_TRIM)
                        
                        # Nếu thực thể sau khi cắt rỗng (ví dụ: chỉ chứa "-"), bỏ qua nó
                        if not trimmed_text:
                            continue

                        # 2. Tìm lại vị trí start mới trong đoạn text gốc
                        # tìm kiếm từ vị trí `start` cũ để tránh tìm nhầm vào một từ giống hệt ở vị trí khác
                        new_start = text.find(trimmed_text, start)
                        
                        # Nếu không tìm thấy (trường hợp hiếm), bỏ qua thực thể này
                        if new_start == -1:
                            continue
                            
                        # 3. Tính toán lại vị trí end
                        new_end = new_start + len(trimmed_text)
                        
                        entities.append((new_start, new_end, label))

                # --- CÁC BƯỚC LÀM SẠCH CŨ VẪN GIỮ NGUYÊN ---
                doc = spacy.blank("en").make_doc(text)
                spans = []
                for start, end, label in entities:
                    span = doc.char_span(start, end, label=label)
                    if span is not None:
                        spans.append(span)

                filtered_spans = filter_spans(spans)
                final_entities = [(span.start_char, span.end_char, span.label_) for span in filtered_spans]

                training_data.append((text, {"entities": final_entities}))
                
        return training_data
    except Exception as e:
        logging.exception(f"An unexpected error occurred while processing {file_path}: {e}")
        return None

def train_ner_model(
    train_data: List[SpacyData], 
    model: Optional[str] = None, 
    output_dir: Optional[pathlib.Path] = None, 
    n_iter: int = 20
) -> spacy.language.Language:
    """
    Trains a spaCy Named Entity Recognition (NER) model.

    Args:
        train_data: The training data in spaCy format.
        model: An existing spaCy model to start from, or None for a blank model.
        output_dir: Directory to save the trained model.
        n_iter: Number of training iterations.

    Returns:
        The trained spaCy Language object.
    """
    if model is not None:
        nlp = spacy.load(model)
        logging.info(f"Loaded model '{model}'")
    else:
        nlp = spacy.blank("en")
        logging.info("Created blank 'en' model")

    # Set up the NER pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Add labels from the training data
    for _, annotations in train_data:
        for ent in annotations.get("entities"): # type: ignore
            ner.add_label(ent[2]) # type: ignore

    # Disable other pipes for focused training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            # Batch up the examples using spaCy's batcher
            for batch in spacy.util.minibatch(train_data, size=spacy.util.compounding(4.0, 32.0, 1.001)): # type: ignore
                examples = []
                for text, annots in batch:
                    examples.append(Example.from_dict(nlp.make_doc(text), annots))
                
                # Update the model
                nlp.update(examples, drop=DROPOUT_RATE, sgd=optimizer, losses=losses)
            
            logging.info(f"Iteration {itn+1}/{n_iter}, Losses: {losses}")

    # Save the trained model
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        nlp.to_disk(output_dir)
        logging.info(f"Saved model to {output_dir}")

    return nlp


def evaluate_model(nlp: spacy.language.Language, test_data: List[SpacyData]):
    """
    Evaluates the NER model and prints a report.

    Args:
        nlp: The trained spaCy model.
        test_data: The test data in spaCy format.
    """
    logging.info("Evaluating model...")
    scorer = spacy.scorer.Scorer() # type: ignore
    examples = []
    for text, annotations in test_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        example.predicted = nlp(str(example.predicted))
        examples.append(example)
    
    scores = scorer.score(examples)
    
    print("\n--- Evaluation Results ---")
    print(f"Overall Precision: {scores['ents_p']:.3f}")
    print(f"Overall Recall: {scores['ents_r']:.3f}")
    print(f"Overall F1-score: {scores['ents_f']:.3f}")
    print("\n--- Scores per Entity Type ---")
    
    # Pretty print per-entity scores
    results_per_type = scores.get('ents_per_type', {})
    if not results_per_type:
        print("No entities found in test data for detailed scoring.")
        return
        
    # Find longest label for alignment
    max_len = max(len(label) for label in results_per_type.keys())
    
    # Header
    print(f"{'TYPE':<{max_len}} | {'P':>6} | {'R':>6} | {'F1':>6}")
    print(f"{'-'*(max_len+1)}+{'-'*8}+{'-'*8}+{'-'*7}")

    for label, metrics in sorted(results_per_type.items()):
        p = metrics.get('p', 0.0)
        r = metrics.get('r', 0.0)
        f = metrics.get('f', 0.0)
        print(f"{label:<{max_len}} | {p:>6.2f} | {r:>6.2f} | {f:>6.2f}")
    print("----------------------------\n")


def save_predictions(nlp: spacy.language.Language, test_data: List[SpacyData], output_dir: pathlib.Path):
    """
    Saves the model's predictions for each test resume to a text file.

    Args:
        nlp: The trained spaCy model.
        test_data: The test data.
        output_dir: Directory to save the prediction files.
    """
    logging.info(f"Saving predictions to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (text, _) in enumerate(test_data):
        doc = nlp(text)
        
        # Group entities by label
        entities_by_label: Dict[str, List[str]] = {}
        for ent in doc.ents:
            entities_by_label.setdefault(ent.label_, []).append(ent.text)
        
        # Write to file
        output_path = output_dir / f"resume_{i+1}_prediction.txt"
        with output_path.open('w', encoding='utf-8') as f:
            f.write(f"--- Predictions for Resume {i+1} ---\n\n")
            if not entities_by_label:
                f.write("No entities found.\n")
                continue
                
            for label, entities in entities_by_label.items():
                f.write(f"## {label}\n")
                # Use set to remove duplicate entities for cleaner output
                for entity in sorted(list(set(entities))):
                    f.write(f"- {entity.strip()}\n")
                f.write("\n")
    logging.info("Finished saving predictions.")


def main():
    """Main function to run the NER training and evaluation pipeline."""
    # 1. Load and convert data
    train_data = convert_dataturks_to_spacy(DATA_PATH)
    test_data = convert_dataturks_to_spacy(TEST_DATA_PATH)

    if not train_data or not test_data:
        logging.error("Could not load training or testing data. Exiting.")
        return

    # 2. Train the model
    nlp = train_ner_model(train_data, output_dir=OUTPUT_DIR, n_iter=N_ITERATIONS)

    # 3. Evaluate the model
    evaluate_model(nlp, test_data)
    
    # 4. Save predictions for manual review
    save_predictions(nlp, test_data, PREDICTIONS_DIR)


if __name__ == "__main__":
    main()