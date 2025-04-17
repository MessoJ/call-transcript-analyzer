import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging # Import logging

# ---to Configure Logging ---

logging.basicConfig(level=logging.WARNING)

#to download nltk.
def download_nltk_data():
    """Downloads required NLTK data if not already present."""
    required_data = [('tokenizers/punkt', 'punkt'), ('corpora/stopwords', 'stopwords')]
    for path, pkg_id in required_data:
        try:
            nltk.data.find(path)
            # print(f"NLTK data '{pkg_id}' found.")
        except nltk.downloader.DownloadError:
            print(f"Downloading NLTK data '{pkg_id}'...")
            nltk.download(pkg_id, quiet=True)

download_nltk_data()

#to initialize VADER sentiment analyzer
try:
    analyzer = SentimentIntensityAnalyzer()
except LookupError:
    print("Downloading VADER lexicon...")
    nltk.download('vader_lexicon', quiet=True)
    analyzer = SentimentIntensityAnalyzer()


# to define speaker patterns and noise words
SPEAKER_PATTERN = re.compile(r"^(Agent|User|Speaker \d+):", re.IGNORECASE | re.MULTILINE)
NOISE_WORDS = set(['uh', 'uhm', 'umm', 'hmm', 'like', 'yeah', 'okay', 'so', 'well'])
# to load English stopwords safely
try:
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
except LookupError:
    print("Stopwords not found initially, attempting download again...")
    download_nltk_data()
    ENGLISH_STOPWORDS = set(stopwords.words('english'))


def count_speaker_turns(transcript: str) -> int:
    """
    Counts the number of speaker turns based on lines starting with speaker identifiers.
    """
    if not isinstance(transcript, str): return 0 #to handle non-string input
    turns = SPEAKER_PATTERN.findall(transcript)
    if not turns and '\n' in transcript.strip():
         # to count non-empty lines as a fallback, crude but better than 1
         return len([line for line in transcript.strip().split('\n') if line.strip()])
    return len(turns) if turns else 1 

def get_utterances(transcript: str) -> list[str]:
    """
    Splits the transcript into a list of utterances based on speaker pattern.
    Keeps only the spoken content, removing the speaker tag. Handles potential leading text.
    """
    if not isinstance(transcript, str): return [] #to handle non-string input

    # to split the transcript by the speaker pattern, keeping the delimiters
    parts = SPEAKER_PATTERN.split(transcript)
    utterances = []

    
    if parts and parts[0] and parts[0].strip():
         utterances.append(parts[0].strip())

    # To iterate through the parts, pairing speaker tags with utterances
    i = 1
    while i < len(parts):
        # to heck if parts[i] is a speaker tag (Agent, User, Speaker X)
        is_speaker_tag = SPEAKER_PATTERN.match(parts[i] + ":")
        if is_speaker_tag and i + 1 < len(parts):
            utterance_text = parts[i+1].strip()
            if utterance_text:
                utterances.append(utterance_text)
            i += 2 
        else:
            
            part_text = parts[i].strip()
            if part_text:
                 utterances.append(part_text)
            i += 1

    
    if len(utterances) <= 1 and transcript.strip():
        
        if utterances and utterances[0] == transcript.strip():
             return utterances
        elif not utterances: 
             return [transcript.strip()]


    # to filter out empty strings just in case
    utterances = [utt for utt in utterances if utt]

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
       
        words_in_utterance = len(utt.split())
        if words_in_utterance > 0: 
             total_words += words_in_utterance
             total_utterances += 1


    return total_words / total_utterances if total_utterances > 0 else 0.0


def tokenize_and_clean(text: str) -> list[str]:
    """Helper function to tokenize text, lowercase, and remove punctuation/stopwords."""
    if not isinstance(text, str): return [] 
    try:
        tokens = word_tokenize(text.lower())
       
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
    if not isinstance(transcript, str): return 0
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
    if not isinstance(transcript, str): return 0.0 
    try:
        vs = analyzer.polarity_scores(transcript)
        return vs['compound']
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return 0.0




def calculate_topic_relevance(transcript: str, expected_topic: str = "default") -> float:
    """
    Placeholder: Calculates how relevant the transcript is to an expected topic.
    Requires topic modeling (e.g., LDA, BERTopic with Gensim/Transformers)
    or text similarity techniques (e.g., sentence embeddings).
    Returns a score (e.g., 0 to 1).
    """
    
    return 0.8 # Placeholder value

def calculate_information_density(transcript: str) -> float:
    """
    Placeholder: Calculates the information density of the transcript.
    Could be based on ratio of content words (nouns, verbs, adj, adv) to total words.
    Requires Part-of-Speech (POS) tagging (e.g., using NLTK or SpaCy).
    Returns a score (e.g., 0 to 1).
    """
    
    return 0.5 # Placeholder value


def analyze_syntax_complexity(transcript: str) -> float:
    """
    Placeholder: Analyzes the syntactic complexity (e.g., avg sentence length, clause depth).
    Requires syntactic parsing (e.g., using SpaCy for dependency trees or NLTK).
    Returns a score representing complexity.
    """
    
    try:
        sentences = nltk.sent_tokenize(transcript)
        if not sentences: return 0.0
        avg_len = sum(len(word_tokenize(s)) for s in sentences) / len(sentences)
        
        complexity_score = min(1.0, max(0.0, (avg_len - 5) / 20.0))
        return complexity_score
    except Exception: # fallback if tokenization fails
         return 0.6 # Placeholder value

def measure_fluency(transcript: str) -> float:
    """
    Placeholder: Measures speech fluency (e.g., based on pause frequency, speech rate).
    Difficult to estimate accurately from text alone. Uses frequency of filler words as proxy.
    Audio analysis would be much better.
    Returns a score representing fluency (higher is better).
    """
    
    if not isinstance(transcript, str) or not transcript.strip(): return 0.0
    total_words = len(transcript.split())
    if total_words == 0: return 1.0 

    filler_count = detect_noise_words(transcript)
    filler_ratio = filler_count / total_words

    
    fluency_score = max(0.0, 1.0 - (filler_ratio * 5))
    return fluency_score

def calculate_error_rate(transcript: str) -> float:
    """
    Placeholder: Calculates the grammatical error rate.
    Requires an external grammar checking tool or API (e.g., LanguageTool-python, GingerIt, or paid APIs).
    This is a complex task and may not be perfectly accurate.
    Returns a ratio (errors / total words or sentences).
    """
    
    return 0.15 # Placeholder value (representing 15% error rate)

def analyze_discourse(transcript: str) -> float:
    """
    Placeholder: Analyzes discourse competence (e.g., coherence, topic management).
    Requires advanced discourse analysis techniques, possibly involving coreference resolution,
    topic segmentation, and analysis of discourse markers. Very complex.
    Returns a score representing competence.
    """
    return 0.6 # Placeholder value (representing ability to discuss familiar topics)

