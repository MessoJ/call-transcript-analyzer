# Helper functions for transcript analysis
import re
import nltk
import os
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

# --- Configure Logging ---
logging.basicConfig(level=logging.WARNING)

# --- to define NLTK Data Directory ---
DATA_DIR = "/tmp/nltk_data"
if not os.path.exists(DATA_DIR):
    try:
        os.makedirs(DATA_DIR)
    except OSError as e:
        print(f"Warning: Could not create NLTK data directory {DATA_DIR}. Error: {e}")
        DATA_DIR = None

# --- to ensure NLTK searches our custom directory first ---
if DATA_DIR and DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, DATA_DIR)

# --- to download necessary NLTK data (run this once) ---
def download_nltk_data(download_dir=None):
    """Downloads required NLTK data if not already present to the specified directory."""
    
    required_data = [('tokenizers/punkt', 'punkt'), ('corpora/stopwords', 'stopwords')]
    for path, pkg_id in required_data:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"NLTK data '{pkg_id}' not found. Downloading to {download_dir or 'default location'}...")
            try:
                nltk.download(pkg_id, download_dir=download_dir, quiet=True)
            except Exception as e:
                print(f"ERROR: Failed to download NLTK data '{pkg_id}'. Error: {e}")


download_nltk_data(download_dir=DATA_DIR)

try:
    analyzer = SentimentIntensityAnalyzer()
except LookupError:
    print("VADER lexicon not found. Downloading...")
    try:
        nltk.download('vader_lexicon', download_dir=DATA_DIR, quiet=True)
        analyzer = SentimentIntensityAnalyzer()
    except Exception as e:
        print(f"ERROR: Failed to download VADER lexicon. Error: {e}")
        analyzer = None

# to define speaker patterns and noise words
SPEAKER_PATTERN = re.compile(r"^(Agent|User|Speaker \d+):", re.IGNORECASE | re.MULTILINE)
NOISE_WORDS = set(['uh', 'uhm', 'umm', 'hmm', 'like', 'yeah', 'okay', 'so', 'well'])
# to load English stopwords safely
try:
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
except LookupError:
    print("Stopwords not found initially, attempting download again...")
    download_nltk_data(download_dir=DATA_DIR) 
    try:
        ENGLISH_STOPWORDS = set(stopwords.words('english'))
    except Exception as e:
        print(f"ERROR: Failed to load stopwords even after download attempt. Error: {e}")
        ENGLISH_STOPWORDS = set()

# --- Helper Functions ---

def regex_word_tokenize(text: str) -> list[str]:
    """Simple regex-based word tokenizer."""
    
    return re.findall(r'\b\w+\b', text.lower())

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
    """Helper function to tokenize text using regex, lowercase, and remove punctuation/stopwords."""
    if not isinstance(text, str): return []
    try:
        
        tokens = regex_word_tokenize(text)
        return [word for word in tokens if word not in ENGLISH_STOPWORDS and len(word) > 1]
    except Exception as e:
        print(f"Error during regex tokenization/cleaning: {e}")
        return []

def count_unique_words(transcript: str) -> int:
    """Counts the number of unique content words (vocabulary size) using regex tokenizer."""
    cleaned_tokens = tokenize_and_clean(transcript)
    return len(set(cleaned_tokens))

def detect_noise_words(transcript: str) -> int:
    """Detects the count of predefined noise/filler words using regex tokenizer."""
    if not isinstance(transcript, str): return 0
    try:
        #to use regex tokenizer
        tokens = regex_word_tokenize(transcript)
        noise_count = sum(1 for token in tokens if token in NOISE_WORDS)
        return noise_count
    except Exception as e:
        print(f"Error during noise word detection with regex: {e}")
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

# ---the  placeholder Functions ---

def calculate_topic_relevance(transcript: str, expected_topic: str = "default") -> float:
    """Placeholder: Calculates topic relevance."""
    return 0.8 # Placeholder value

def calculate_information_density(transcript: str) -> float:
    """Placeholder: Calculates information density."""
    
    return 0.5 # Placeholder value

def analyze_syntax_complexity(transcript: str) -> float:
    """Analyzes syntax complexity using avg sentence length. Tries NLTK sent_tokenize, falls back to regex."""
    if not isinstance(transcript, str) or not transcript.strip(): return 0.0
    sentences = []
    try:
       
        sentences = sent_tokenize(transcript)
    except LookupError:
        print("Warning: NLTK 'punkt' data missing for sentence tokenization. Falling back to basic regex split.")
        
        sentences = re.split(r'[.?!]\s+', transcript)
        sentences = [s for s in sentences if s] 
    except Exception as e:
         print(f"Error during NLTK sentence tokenization: {e}. Falling back to basic regex split.")
         sentences = re.split(r'[.?!]\s+', transcript)
         sentences = [s for s in sentences if s] 

    if not sentences: return 0.0

    total_words, valid_sentences = 0, 0
    for s in sentences:
        try:
            
            words = regex_word_tokenize(s)
            if words:
                total_words += len(words)
                valid_sentences += 1
        except Exception as word_e:
             print(f"Error tokenizing words in sentence with regex: {s[:50]}... Error: {word_e}")
             continue

    if valid_sentences == 0: return 0.0
    avg_len = total_words / valid_sentences
    # Normalize score 
    complexity_score = min(1.0, max(0.0, (avg_len - 5) / 20.0))
    return complexity_score


def measure_fluency(transcript: str) -> float:
    """Measures fluency using filler word ratio (uses regex tokenizer via detect_noise_words)."""
    if not isinstance(transcript, str) or not transcript.strip(): return 0.0
    total_words = len(transcript.split()) 
    if total_words == 0: return 1.0
   
    filler_count = detect_noise_words(transcript)
    filler_ratio = filler_count / total_words
    fluency_score = max(0.0, 1.0 - (filler_ratio * 5))
    return fluency_score

def calculate_error_rate(transcript: str) -> float:
    """Placeholder: Calculates grammatical error rate."""
    # this requires external tool like language_tool_python and Java.
    return 0.15 # Placeholder value

def analyze_discourse(transcript: str) -> float:
    """Placeholder: Analyzes discourse competence."""
    return 0.6 # Placeholder value
