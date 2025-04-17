# src/utils.py
# Helper functions for transcript analysis
# Includes implementations for some functions using NLTK and VADER.

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging # Import logging

# --- Configure Logging ---
# Configure logging to suppress unnecessary warnings or info from libraries
logging.basicConfig(level=logging.WARNING) # Set default level to WARNING
# You can adjust the level for specific libraries if needed, e.g.:
# logging.getLogger('nltk').setLevel(logging.WARNING)

# --- Download necessary NLTK data (run this once) ---
# Use a function to handle downloads cleanly
def download_nltk_data():
    """Downloads required NLTK data if not already present."""
    required_data = [('tokenizers/punkt', 'punkt'), ('corpora/stopwords', 'stopwords')]
    for path, pkg_id in required_data:
        try:
            nltk.data.find(path)
            # print(f"NLTK data '{pkg_id}' found.") # Optional: confirmation message
        # Corrected: Catch LookupError instead of nltk.downloader.DownloadError
        except LookupError:
            print(f"NLTK data '{pkg_id}' not found. Downloading...")
            nltk.download(pkg_id, quiet=True) # Download quietly

download_nltk_data() # Call the download function on import
# -----------------------------------------------------

# Initialize VADER sentiment analyzer
try:
    analyzer = SentimentIntensityAnalyzer()
except LookupError:
    print("Downloading VADER lexicon...")
    nltk.download('vader_lexicon', quiet=True)
    analyzer = SentimentIntensityAnalyzer()


# Define speaker patterns and noise words
SPEAKER_PATTERN = re.compile(r"^(Agent|User|Speaker \d+):", re.IGNORECASE | re.MULTILINE)
# Basic list of potential noise/filler words
NOISE_WORDS = set(['uh', 'uhm', 'umm', 'hmm', 'like', 'yeah', 'okay', 'so', 'well'])
# Load English stopwords safely
try:
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
except LookupError:
    print("Stopwords not found initially, attempting download again...")
    download_nltk_data() # Ensure stopwords are downloaded
    ENGLISH_STOPWORDS = set(stopwords.words('english'))


def count_speaker_turns(transcript: str) -> int:
    """
    Counts the number of speaker turns based on lines starting with speaker identifiers.
    """
    if not isinstance(transcript, str): return 0 # Handle non-string input
    turns = SPEAKER_PATTERN.findall(transcript)
    # If no standard pattern matches, check if there are line breaks (simple turn indicator)
    if not turns and '\n' in transcript.strip():
         # Count non-empty lines as a fallback, crude but better than 1
         return len([line for line in transcript.strip().split('\n') if line.strip()])
    return len(turns) if turns else 1 # Assume at least one speaker if no pattern/newlines

def get_utterances(transcript: str) -> list[str]:
    """
    Splits the transcript into a list of utterances based on speaker pattern.
    Keeps only the spoken content, removing the speaker tag. Handles potential leading text.
    """
    if not isinstance(transcript, str): return [] # Handle non-string input

    # Split the transcript by the speaker pattern. Use finditer to get matches and split points.
    utterances = []
    last_end = 0
    for match in SPEAKER_PATTERN.finditer(transcript):
        start, end = match.span()
        # Add text between the last match and this one (or from the beginning)
        if start > last_end:
            utterances.append(transcript[last_end:start].strip())
        # The matched speaker tag itself is not added.
        last_end = end

    # Add any remaining text after the last match
    if last_end < len(transcript):
        utterances.append(transcript[last_end:].strip())

    # Filter out empty strings that might result from consecutive newlines etc.
    utterances = [utt for utt in utterances if utt]

    # If splitting produced no results (e.g., transcript has no speaker tags), treat the whole thing as one utterance
    if not utterances and transcript.strip():
        return [transcript.strip()]

    return utterances


def calculate_avg_utterance_length(transcript: str) -> float:
    """
    Calculates the average length (in words) of utterances in the transcript.
    """
    utterances = get_utterances(transcript)
    if not utterances:
        return 0.0

    total_words = 0
    total_utterances = 0
    for utt in utterances:
        # Simple word count by splitting whitespace
        words_in_utterance = len(utt.split())
        if words_in_utterance > 0: # Only count utterances with words
             total_words += words_in_utterance
             total_utterances += 1


    return total_words / total_utterances if total_utterances > 0 else 0.0


def tokenize_and_clean(text: str) -> list[str]:
    """Helper function to tokenize text, lowercase, and remove punctuation/stopwords."""
    if not isinstance(text, str): return [] # Handle non-string input
    try:
        tokens = word_tokenize(text.lower())
        # Keep alphanumeric tokens and remove stopwords/single chars
        return [word for word in tokens if word.isalnum() and word not in ENGLISH_STOPWORDS and len(word) > 1]
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return []


def count_unique_words(transcript: str) -> int:
    """
    Counts the number of unique content words (vocabulary size) using NLTK.
    Excludes common stopwords and punctuation.
    """
    cleaned_tokens = tokenize_and_clean(transcript)
    return len(set(cleaned_tokens))


def detect_noise_words(transcript: str) -> int:
    """
    Detects the count of predefined noise/filler words.
    Uses simple case-insensitive matching on tokens.
    """
    if not isinstance(transcript, str): return 0 # Handle non-string input
    try:
        tokens = word_tokenize(transcript.lower())
        noise_count = sum(1 for token in tokens if token in NOISE_WORDS)
        return noise_count
    except Exception as e:
        print(f"Error during noise word detection: {e}")
        return 0


def analyze_sentiment(transcript: str) -> float:
    """
    Analyzes the overall sentiment of the transcript using VADER.
    Returns the compound score (ranging from -1 to 1).
    """
    if not isinstance(transcript, str): return 0.0 # Handle non-string input
    try:
        vs = analyzer.polarity_scores(transcript)
        return vs['compound']
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return 0.0


# --- Placeholder Functions (Require more advanced implementation) ---

def calculate_topic_relevance(transcript: str, expected_topic: str = "default") -> float:
    """
    Placeholder: Calculates how relevant the transcript is to an expected topic.
    Requires topic modeling (e.g., LDA, BERTopic with Gensim/Transformers)
    or text similarity techniques (e.g., sentence embeddings).
    Returns a score (e.g., 0 to 1).
    """
    # print("Warning: calculate_topic_relevance is a placeholder.") # Keep commented unless debugging
    # Implementation idea:
    # 1. Generate embeddings for the transcript and the expected topic description (e.g., using sentence-transformers).
    # 2. Calculate cosine similarity between embeddings.
    # Requires: pip install sentence-transformers
    return 0.8 # Placeholder value

def calculate_information_density(transcript: str) -> float:
    """
    Placeholder: Calculates the information density of the transcript.
    Could be based on ratio of content words (nouns, verbs, adj, adv) to total words.
    Requires Part-of-Speech (POS) tagging (e.g., using NLTK or SpaCy).
    Returns a score (e.g., 0 to 1).
    """
    # print("Warning: calculate_information_density is a placeholder.") # Keep commented unless debugging
    # Implementation idea:
    # 1. Perform POS tagging on the transcript tokens (nltk.pos_tag(word_tokenize(transcript))).
    # 2. Count content words (NN*, VB*, JJ*, RB* tags).
    # 3. Calculate ratio: content_words / total_words.
    # Requires NLTK POS tagger data: nltk.download('averaged_perceptron_tagger')
    return 0.5 # Placeholder value


def analyze_syntax_complexity(transcript: str) -> float:
    """
    Placeholder: Analyzes the syntactic complexity (e.g., avg sentence length, clause depth).
    Requires syntactic parsing (e.g., using SpaCy for dependency trees or NLTK).
    Returns a score representing complexity.
    """
    # print("Warning: analyze_syntax_complexity is a placeholder.") # Keep commented unless debugging
    # Implementation idea:
    # 1. Use SpaCy to parse the text into sentences (doc.sents).
    # 2. Calculate average sentence length (in words).
    # 3. Analyze dependency tree depth for each sentence (more complex).
    # Requires SpaCy and a model: pip install spacy; python -m spacy download en_core_web_sm
    # Simple approximation: average sentence length
    try:
        sentences = nltk.sent_tokenize(transcript)
        if not sentences: return 0.0
        # Ensure punkt tokenizer is available for sentence and word tokenization
        download_nltk_data() # Double check NLTK data needed here
        avg_len = sum(len(word_tokenize(s)) for s in sentences) / len(sentences)
        # Normalize score (e.g., map 5-25 words/sentence to 0-1)
        complexity_score = min(1.0, max(0.0, (avg_len - 5) / 20.0))
        return complexity_score
    except Exception as e: # Fallback if tokenization fails
         print(f"Error during syntax complexity analysis: {e}")
         return 0.6 # Placeholder value

def measure_fluency(transcript: str) -> float:
    """
    Placeholder: Measures speech fluency (e.g., based on pause frequency, speech rate).
    Difficult to estimate accurately from text alone. Uses frequency of filler words as proxy.
    Audio analysis would be much better.
    Returns a score representing fluency (higher is better).
    """
    # print("Warning: measure_fluency is a placeholder.") # Keep commented unless debugging
    # Implementation idea (text-based approximation):
    # 1. Calculate ratio of filler words (from detect_noise_words) to total words.
    # 2. Invert/normalize the score (lower filler ratio -> higher fluency).
    if not isinstance(transcript, str) or not transcript.strip(): return 0.0
    total_words = len(transcript.split())
    if total_words == 0: return 1.0 # Perfect fluency if empty? Or 0? Let's say 1.0

    filler_count = detect_noise_words(transcript)
    filler_ratio = filler_count / total_words

    # Crude example: Score decreases as filler ratio increases. Max score 1.0.
    # Scaled so that 10% filler ratio gives ~0.5 score.
    fluency_score = max(0.0, 1.0 - (filler_ratio * 5))
    return fluency_score

def calculate_error_rate(transcript: str) -> float:
    """
    Placeholder: Calculates the grammatical error rate.
    Requires an external grammar checking tool or API (e.g., LanguageTool-python, GingerIt, or paid APIs).
    This is a complex task and may not be perfectly accurate.
    Returns a ratio (errors / total words or sentences).
    """
    # print("Warning: calculate_error_rate is a placeholder.") # Keep commented unless debugging
    # Implementation idea:
    # 1. Integrate a library like language_tool_python (requires Java runtime).
    # 2. Initialize LanguageTool: tool = language_tool_python.LanguageTool('en-US')
    # 3. Find matches (errors): matches = tool.check(transcript)
    # 4. Calculate error rate: len(matches) / len(word_tokenize(transcript))
    # Requires: pip install language_tool_python
    return 0.15 # Placeholder value (representing 15% error rate)

def analyze_discourse(transcript: str) -> float:
    """
    Placeholder: Analyzes discourse competence (e.g., coherence, topic management).
    Requires advanced discourse analysis techniques, possibly involving coreference resolution,
    topic segmentation, and analysis of discourse markers. Very complex.
    Returns a score representing competence.
    """
    # print("Warning: analyze_discourse is a placeholder.") # Keep commented unless debugging
    # Implementation idea:
    # 1. Use coreference resolution (e.g., with SpaCy/neuralcoref or Hugging Face).
    # 2. Analyze topic shifts (e.g., using embeddings or topic modeling on segments).
    # 3. Analyze use of discourse markers ('however', 'therefore', 'so', 'because').
    return 0.6 # Placeholder value (representing ability to discuss familiar topics)

