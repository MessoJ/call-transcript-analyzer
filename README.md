
# Call Transcript Analyzer

This project provides tools for analyzing call transcripts using Natural Language Processing (NLP) techniques. It includes modules for:

1.  **Information Extraction:** Identifying key entities, topics, sentiment, intent, and phrases from transcripts. (Note: Specific implementation code for this section was not provided in the source document).
2.  **Quality Assessment:** Evaluating transcripts based on metrics like length, utterance length, noise, sentiment, relevance, speaker turns, and information density to determine suitability for further analysis.
3.  **English Proficiency Assessment:** Estimating the speaker's English proficiency level (aligned with CEFR) based on lexical, syntactic, fluency, error, and pragmatic features.

## Setup

1.  **Clone the repository**
    cd call_transcript_analyzer
    

2.  **Create a virtual environment**

3.  **Install dependencies:**
    * Install system dependencies i.e. Graphviz
    * Install Python packages:
        ```bash
        pip install -r requirements.txt
        ```
      * Download models for SpaCy or Hugging Face Transformers 

## Usage

run directly from ``python main.py`` o
or

Import the functions from the `src` modules into your Python scripts or notebooks.

```python
from src.quality_assessment import assess_call_quality
from src.proficiency_assessment import assess_english_level

# Assess quality
quality_result = assess_call_quality(transcript)
print("Quality Assessment:", quality_result)

# Assess proficiency
proficiency_level = assess_english_level(transcript)
print("Estimated Proficiency Level:", proficiency_level)


