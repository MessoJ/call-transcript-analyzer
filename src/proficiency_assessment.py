from .utils import (
    count_unique_words,
    analyze_syntax_complexity,
    measure_fluency,
    calculate_error_rate,
    analyze_discourse
)

def assess_english_level(transcript: str) -> str:
    """
    Assesses the estimated English proficiency level based on CEFR-aligned criteria.

    Args:
        transcript: The call transcript text.

    Returns:
        A string representing the estimated CEFR level (e.g., "A1", "A2", "B1", "B2", "C1", "C2").
    """
    if not transcript or not isinstance(transcript, str):
        print("Warning: Invalid or empty transcript provided for proficiency assessment.")
        return "Unknown"

    # 1. To extract key features using helper functions.
    metrics = {
        "vocabulary_size": count_unique_words(transcript),
        "sentence_complexity": analyze_syntax_complexity(transcript), # Placeholder returns a score
        "fluency_score": measure_fluency(transcript), # Placeholder returns a score
        "error_rate": calculate_error_rate(transcript), # Placeholder returns a ratio
        "discourse_competence": analyze_discourse(transcript) # Placeholder returns a score
    }

    # 2. Determine CEFR level based on thresholds (simplified from the table)

    vocab = metrics["vocabulary_size"]
    error = metrics["error_rate"]

    if vocab <= 500 and error >= 0.30:
        return "A1"
    elif 500 < vocab <= 1000 and 0.20 <= error < 0.30:
        return "A2"
    elif 1000 < vocab <= 2000 and 0.10 <= error < 0.20:
        return "B1"
    elif 2000 < vocab <= 3500 and 0.05 <= error < 0.10:
        return "B2"
    elif 3500 < vocab <= 5000 and error < 0.05:
         # Assuming low error rate for C1 as well
        return "C1"
    elif vocab > 5000 and error < 0.05: # Assuming very low error rate for C2
        return "C2"
    else:
        # for fallback
        if vocab > 5000: return "C2"
        if vocab > 3500: return "C1"
        if vocab > 2000: return "B2"
        if vocab > 1000: return "B1"
        if vocab > 500: return "A2"
        return "A1" 


# Test usage 
if __name__ == '__main__':
    sample_transcript_b1 = """
    User: Hello. Yes, I want ask about my, uh, internet connection. It is slow today.
    Agent: Okay, I can help check that. Can you tell me your account number?
    User: Yes, it is one two three seven five. Yesterday it was working good, but now... very slow.
    Agent: I see. Let me check the status for you. It might be maintenance in your area.
    User: Ah, okay. How long this maintenance take?
    """
    level = assess_english_level(sample_transcript_b1)
    print(f"Sample Transcript Estimated Level: {level}") # Expected: B1 (based on placeholder values)

    sample_transcript_c1 = """
    User: Good morning. I'm calling to inquire about the feasibility of upgrading my current broadband package.
    User: Specifically, I'm interested in understanding the potential throughput increase and associated costs with the fiber-optic options you offer.
    Agent: Certainly. I can provide detailed information on our fiber packages. Could you please confirm your address so I can check availability and specific offerings?
    User: Of course. The address is 210 Colny Street. I've noticed some latency during peak hours recently, and I'm hoping a fiber upgrade would mitigate that significantly.
    Agent: That's a common reason for upgrading. Fiber typically offers much lower latency and more consistent speeds. Let me pull up the options for 210 Colny Street right now.
    """
    level_c1 = assess_english_level(sample_transcript_c1)
    print(f"Sample Transcript 2 Estimated Level: {level_c1}") # Expected: C1/C2 (based on placeholder values)
