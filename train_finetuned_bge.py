import pathlib
import logging
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch # Import torch to check for GPU

# --- 1. Configuration ---
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# --- REPRODUCIBILITY SETUP (Fixes the "Friend's Run" Issue) ---
def set_seed(seed=42):
    """
    Sets the seed for all random number generators to ensure 
    reproducible results on CPU and GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Force strict determinism for NVIDIA GPUs (School GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logging.info(f"🔒 Random seed set to {seed} for reproducibility.")

# !!! CRITICAL: Call this before doing anything else !!!
set_seed(42)

# Start from the pre-trained BGE model
BASE_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Where your new, fine-tuned model will be saved
NEW_MODEL_NAME = "./bge-finetuned-all-domains"

# The 4 MTRAG domains to train on
MTRAG_DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]

# Training parameters
TRAIN_BATCH_SIZE = 16 # 16 or 32 is a good batch size. Lower to 8 if you get "Out of Memory" errors.
EPOCHS = 1 # 1 Chosen for the fast proof-of-concept. Increase to 3-5 for a better model.
WARMUP_STEPS = 100

def load_mtrag_examples(domain):
    """
    Loads a single MTRAG domain and converts it into
    (query, positive_passage) InputExamples.
    """
    logging.info(f"Loading MTRAG domain: {domain}...")
    data_root = pathlib.Path(".")
    corpus_file = data_root / "corpora" / "passage_level" / f"{domain}.jsonl"
    query_file = data_root / "human" / "retrieval_tasks" / domain / f"{domain}_questions.jsonl"
    qrels_file = data_root / "human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
    
    try:
        corpus, queries, qrels = GenericDataLoader(
            corpus_file=str(corpus_file),
            query_file=str(query_file),
            qrels_file=str(qrels_file)
        ).load_custom()
    except Exception as e:
        logging.error(f"Error loading {domain}: {e}. Skipping.")
        return []

    domain_examples = []
    for query_id, doc_infos in qrels.items():
        query_text = queries.get(query_id)
        if not query_text:
            continue
            
        for doc_id, score in doc_infos.items():
            if score > 0: # score > 0 means it's a relevant (positive) passage
                doc = corpus.get(doc_id)
                if doc:
                    # Combine title and text for richer context
                    doc_text = doc.get("title", "") + " " + doc.get("text", "")
                    domain_examples.append(InputExample(texts=[query_text, doc_text]))
    
    logging.info(f"Loaded {len(domain_examples)} examples from {domain}")
    return domain_examples

def load_custom_examples():
    """
    #####################################################################
    # TODO: This is your placeholder function!
    #
    # Replace this with code to load your own dataset.
    # It MUST return a list of InputExample objects, where each object
    # is a (query, positive_passage) pair.
    #####################################################################
    """
    logging.info("Loading custom dataset...")
    my_examples = []
    
    # --- Start of Placeholder ---
    # As you haven't decided on data, this just has two examples.
    # When you're ready, you'll replace this part.
    my_examples.append(InputExample(texts=[
        "What are the side effects of lisinopril?",
        "Lisinopril, an ACE inhibitor, can cause a persistent dry cough, dizziness, and headache."
    ]))
    my_examples.append(InputExample(texts=[
        "How does photosynthesis work?",
        "Photosynthesis is the process used by plants to convert light energy into chemical energy."
    ]))
    # --- End of Placeholder ---
    
    logging.info(f"Loaded {len(my_examples)} custom examples (from placeholder)")
    return my_examples

# --- 2. Main Training Logic ---
def run_training():
    logging.info("Starting training process...")
    
    # Check for GPU
    if not torch.cuda.is_available():
        logging.error("CUDA (NVIDIA GPU) not available. Training will be extremely slow.")
        logging.error("Please ensure you are in the correct conda environment with PyTorch and CUDA installed.")
        # We can still proceed, but it will be on CPU and very slow.
        
    all_train_examples = []

    # Load all 4 MTRAG domains
    for domain in MTRAG_DOMAINS:
        all_train_examples.extend(load_mtrag_examples(domain))

    # Load your custom dataset
    all_train_examples.extend(load_custom_examples())

    logging.info(f"Total training examples: {len(all_train_examples)}")

    if not all_train_examples:
        logging.error("No training examples found. Exiting.")
        return

    # --- 3. Setup Model and Dataloader ---
    logging.info(f"Loading base model: {BASE_MODEL_NAME}...")
    # This will automatically use the GPU if available
    model = SentenceTransformer(BASE_MODEL_NAME)

    # Dataloader for the training examples
    train_dataloader = DataLoader(all_train_examples, shuffle=True, batch_size=TRAIN_BATCH_SIZE)

    # --- 4. Define Loss Function (The "Learning" Part) ---
    # This is the "contrastive learning" loss.
    # It automatically uses other examples in the batch as "negatives."
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # --- 5. Start Training ---
    logging.info(f"Starting model fine-tuning... (Epochs={EPOCHS}, Batch Size={TRAIN_BATCH_SIZE})")
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=EPOCHS,
              warmup_steps=WARMUP_STEPS,
              output_path=NEW_MODEL_NAME,
              show_progress_bar=True,
              checkpoint_save_steps=1000,
              checkpoint_path=f"{NEW_MODEL_NAME}-checkpoints"
             )

    logging.info(f"Training complete. New model saved to: {NEW_MODEL_NAME}")

if __name__ == "__main__":
    run_training()