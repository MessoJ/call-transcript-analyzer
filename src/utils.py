# src/utils.py
# Helper functions for transcript analysis
# Includes implementations for some functions using NLTK and VADER.

import re
import nltk
import os # Import os module
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

# --- Configure Logging ---
logging.basicConfig(level=logging.WARNING)
# logging.getLogger('nltk').setLevel(logging.INFO) # Uncomment for more NLTK download details if needed

# --- Define NLTK Data Directory ---
# Use /tmp/ as it's often writable in restricted environments
DATA_DIR = "/tmp/nltk_data"
if not os.path.exists(DATA_DIR):
    try:
        os.makedirs(DATA_DIR)
    except OSError as e:
        print(f"Warning: Could not create NLTK data directory {DATA_DIR}. Error: {e}")
        # Fallback to default behavior if directory creation fails
        DATA_DIR = None

# --- Ensure NLTK searches our custom directory first ---
if DATA_DIR and DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, DATA_DIR)
    # print(f"NLTK data path set to: {nltk.data.path}") # Uncomment for debugging path issues

# --- Download necessary NLTK data (run this once) ---
def download_nltk_data(download_dir=None):
    """Downloads required NLTK data if not already present to the specified directory."""
    required_data = [('tokenizers/punkt', 'punkt'), ('corpora/stopwords', 'stopwords')]
    for path, pkg_id in required_data:
        try:
            # Check if found in *any* path known to NLTK
            nltk.data.find(path)
            # print(f"NLTK data '{pkg_id}' found.") # Optional
        except LookupError:
            print(f"NLTK data '{pkg_id}' not found. Downloading to {download_dir or 'default location'}...")
            try:
                # Download specifically to our directory if provided
                nltk.download(pkg_id, download_dir=download_dir, quiet=True)
            except Exception as e:
                print(f"ERROR: Failed to download NLTK data '{pkg_id}'. Error: {e}")
                print("Please check network connection, permissions, and NLTK setup.")

# Call download function once at module load, specifying the directory
download_nltk_data(download_dir=DATA_DIR)
# -----------------------------------------------------

# Initialize VADER sentiment analyzer
# VADER lexicon also needs to be findable
try:
    analyzer = SentimentIntensityAnalyzer()
except LookupError:
    print("VADER lexicon not found. Downloading...")
    try:
        # Download vader_lexicon, respecting the DATA_DIR if set
        nltk.download('vader_lexicon', download_dir=DATA_DIR, quiet=True)
        analyzer = SentimentIntensityAnalyzer()
    except Exception as e:
        print(f"ERROR: Failed to download VADER lexicon. Error: {e}")
        analyzer = None


# Define speaker patterns and noise words
SPEAKER_PATTERN = re.compile(r"^(Agent|User|Speaker \d+):", re.IGNORECASE | re.MULTILINE)
NOISE_WORDS = set(['uh', 'uhm', 'umm', 'hmm', 'like', 'yeah', 'okay', 'so', 'well'])
# Load English stopwords safely
try:
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
except LookupError:
    print("Stopwords not found initially, attempting download again...")
    download_nltk_data(download_dir=DATA_DIR) # Ensure stopwords are downloaded
    try:
        ENGLISH_STOPWORDS = set(stopwords.words('english'))
    except Exception as e:
        print(f"ERROR: Failed to load stopwords even after download attempt. Error: {e}")
        ENGLISH_STOPWORDS = set()


# --- Helper Functions ---

def count_speaker_turns(transcript: str) -> int:
    """Counts the number of speaker turns based on lines starting with speaker identifiers."""
    if not isinstance(transcript, str): return 0
    turns = SPEAKER_PATTERN.findall(transcript)
    if not turns and '\n' in transcript.strip():
         return len([line for line in transcript.strip().split('\n') if line.strip()])
    return len(turns) if turns else 1

def get_utterances(transcript: str) -> list[str]:
    """Splits the transcript into a list of utterances based on speaker pattern."""
    if not isinstance(transcript, str): return []
    utterances = []
    last_end = 0
    try:
        for match in SPEAKER_PATTERN.finditer(transcript):
            start, end = match.span()
            if start > last_end: utterances.append(transcript[last_end:start].strip())
            last_end = end
        if last_end < len(transcript): utterances.append(transcript[last_end:].strip())
        utterances = [utt for utt in utterances if utt]
        if not utterances and transcript.strip(): return [transcript.strip()]
        return utterances
    except Exception as e:
        print(f"Error during utterance splitting: {e}")
        return [transcript.strip()] if transcript and isinstance(transcript, str) else []

def calculate_avg_utterance_length(transcript: str) -> float:
    """Calculates the average length (in words) of utterances in the transcript."""
    utterances = get_utterances(transcript)
    if not utterances: return 0.0
    total_words, total_utterances = 0, 0
    for utt in utterances:
        words_in_utterance = len(utt.split())
        if words_in_utterance > 0:
             total_words += words_in_utterance
             total_utterances += 1
    return total_words / total_utterances if total_utterances > 0 else 0.0

def tokenize_and_clean(text: str) -> list[str]:
    """Helper function to tokenize text, lowercase, and remove punctuation/stopwords."""
    if not isinstance(text, str): return []
    try:
        # No need to call download_nltk_data() here anymore
        tokens = word_tokenize(text.lower())
        return [word for word in tokens if word.isalnum() and word not in ENGLISH_STOPWORDS and len(word) > 1]
    except LookupError:
        print("ERROR: NLTK 'punkt' tokenizer data not found during word_tokenize. Ensure NLTK data path is correct and data downloaded.")
        # Attempt one more download just in case, before failing
        download_nltk_data(download_dir=DATA_DIR)
        try:
             tokens = word_tokenize(text.lower())
             return [word for word in tokens if word.isalnum() and word not in ENGLISH_STOPWORDS and len(word) > 1]
        except Exception:
             return [] # Return empty list if still fails
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return []

def count_unique_words(transcript: str) -> int:
    """Counts the number of unique content words (vocabulary size) using NLTK."""
    cleaned_tokens = tokenize_and_clean(transcript)
    return len(set(cleaned_tokens))

def detect_noise_words(transcript: str) -> int:
    """Detects the count of predefined noise/filler words."""
    if not isinstance(transcript, str): return 0
    try:
        # No need to call download_nltk_data() here anymore
        tokens = word_tokenize(transcript.lower())
        noise_count = sum(1 for token in tokens if token in NOISE_WORDS)
        return noise_count
    except LookupError:
        print("ERROR: NLTK 'punkt' tokenizer data not found during noise word detection. Ensure NLTK data path is correct and data downloaded.")
        # Attempt one more download just in case, before failing
        download_nltk_data(download_dir=DATA_DIR)
        try:
            tokens = word_tokenize(transcript.lower())
            return sum(1 for token in tokens if token in NOISE_WORDS)
        except Exception:
            return 0 # Return 0 if still fails
    except Exception as e:
        print(f"Error during noise word detection: {e}")
        return 0

def analyze_sentiment(transcript: str) -> float:
    """Analyzes the overall sentiment of the transcript using VADER."""
    if not isinstance(transcript, str): return 0.0
    if analyzer is None:
         print("ERROR: VADER sentiment analyzer not initialized correctly.")
         return 0.0
    try:
        vs = analyzer.polarity_scores(transcript)
        return vs['compound']
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return 0.0

# --- Placeholder Functions ---

def calculate_topic_relevance(transcript: str, expected_topic: str = "default") -> float:
    """Placeholder: Calculates topic relevance."""
    return 0.8 # Placeholder value

def calculate_information_density(transcript: str) -> float:
    """Placeholder: Calculates information density."""
    # Requires: nltk.download('averaged_perceptron_tagger', download_dir=DATA_DIR)
    # try:
    #     tokens = word_tokenize(transcript)
    #     tagged_tokens = nltk.pos_tag(tokens)
    #     # ... rest of calculation ...
    # except LookupError:
    #      print("ERROR: NLTK data missing for information density (punkt or tagger).")
    # except Exception as e:
    #      print(f"Error during information density calc: {e}")
    return 0.5 # Placeholder value

def analyze_syntax_complexity(transcript: str) -> float:
    """Placeholder: Analyzes syntax complexity using avg sentence length."""
    if not isinstance(transcript, str) or not transcript.strip(): return 0.0
    try:
        # No need to call download_nltk_data() here anymore
        sentences = sent_tokenize(transcript)
        if not sentences: return 0.0
        total_words, valid_sentences = 0, 0
        for s in sentences:
            try:
                words = word_tokenize(s)
                if words:
                    total_words += len(words)
                    valid_sentences += 1
            except LookupError:
                 print("ERROR: NLTK 'punkt' data missing for word tokenization within sentence complexity.")
                 continue # Skip sentence
            except Exception as word_e:
                 print(f"Error tokenizing words in sentence: {s[:50]}... Error: {word_e}")
                 continue

        if valid_sentences == 0: return 0.0
        avg_len = total_words / valid_sentences
        complexity_score = min(1.0, max(0.0, (avg_len - 5) / 20.0))
        return complexity_score
    except LookupError:
        print("ERROR: NLTK 'punkt' data missing for sentence tokenization. Cannot calculate syntax complexity.")
        # Attempt one more download just in case, before failing
        download_nltk_data(download_dir=DATA_DIR)
        try:
             # Retry the whole calculation
             sentences = sent_tokenize(transcript)
             # ... (repeat calculation logic - omitted for brevity, ideally refactor) ...
             return 0.6 # Placeholder if retry logic is complex
        except Exception:
             return 0.6 # Return placeholder if still fails
    except Exception as e:
         print(f"Error during syntax complexity analysis: {e}")
         return 0.6

def measure_fluency(transcript: str) -> float:
    """Placeholder: Measures fluency using filler word ratio."""
    if not isinstance(transcript, str) or not transcript.strip(): return 0.0
    total_words = len(transcript.split())
    if total_words == 0: return 1.0
    filler_count = detect_noise_words(transcript) # Uses updated detect_noise_words
    filler_ratio = filler_count / total_words
    fluency_score = max(0.0, 1.0 - (filler_ratio * 5))
    return fluency_score

def calculate_error_rate(transcript: str) -> float:
    """Placeholder: Calculates grammatical error rate."""
    # Requires external tool like language_tool_python and potentially Java.
    # Requires: nltk.download('punkt', download_dir=DATA_DIR) if using word_tokenize here.
    return 0.15 # Placeholder value

def analyze_discourse(transcript: str) -> float:
    """Placeholder: Analyzes discourse competence."""
    return 0.6 # Placeholder value

