from .utils import (
    calculate_avg_utterance_length,
    detect_noise_words,
    analyze_sentiment,
    calculate_topic_relevance,
    count_speaker_turns,
    calculate_information_density
)

def assess_call_quality(transcript: str) -> dict:
    """
    Evaluates if a call transcript is suitable for further processing based on various metrics.

    Args:
        transcript: The call transcript text.

    Returns:
        A dictionary containing the quality assessment results:
        {
            "is_good_quality": bool,
            "quality_score": int,
            "quality_ratio": float,
            "metrics": dict,
            "rejection_reasons": list[str]
        }
    """
    if not transcript or not isinstance(transcript, str):
        return {
            "is_good_quality": False,
            "quality_score": 0,
            "quality_ratio": 0.0,
            "metrics": {},
            "rejection_reasons": ["Invalid or empty transcript provided."]
        }

    words = transcript.split()
    word_count = len(words)

    # 1. To calculate basic metrics using helper functions
    metrics = {
        "transcript_length": word_count,
        "avg_utterance_length": calculate_avg_utterance_length(transcript),
        "noise_ratio": detect_noise_words(transcript) / word_count if word_count > 0 else 1.0,
        "sentiment_score": analyze_sentiment(transcript),
        "topic_relevance": calculate_topic_relevance(transcript), # Assumes a default topic relevance check
        "speaker_turns": count_speaker_turns(transcript),
        "information_density": calculate_information_density(transcript)
    }

    # 2. To define quality thresholds
    thresholds = {
        "min_transcript_length": 50,
        "min_avg_utterance_length": 5,
        "max_noise_ratio": 0.3,
        "min_sentiment_score": -0.8, # To allow slightly negative sentiment
        "min_topic_relevance": 0.6,
        "min_speaker_turns": 3,
        "min_information_density": 0.4
    }

    # 3.To calculate quality score based on thresholds
    quality_score = 0
    max_possible_score = len(thresholds)
    reasons = []

    # To check each metric against its threshold
    if metrics["transcript_length"] >= thresholds["min_transcript_length"]:
        quality_score += 1
    else:
        reasons.append(f"Transcript too short ({metrics['transcript_length']} words, minimum {thresholds['min_transcript_length']})")

    if metrics["avg_utterance_length"] >= thresholds["min_avg_utterance_length"]:
        quality_score += 1
    else:
        reasons.append(f"Utterances too short (avg {metrics['avg_utterance_length']:.2f} words, minimum {thresholds['min_avg_utterance_length']})")

    if metrics["noise_ratio"] <= thresholds["max_noise_ratio"]: 
        quality_score += 1
    else:
        reasons.append(f"Too much noise ({metrics['noise_ratio']:.2f} ratio, maximum {thresholds['max_noise_ratio']})")

    if metrics["sentiment_score"] >= thresholds["min_sentiment_score"]:
        quality_score += 1
    else:
        reasons.append(f"Extremely negative sentiment ({metrics['sentiment_score']:.2f}, minimum {thresholds['min_sentiment_score']})")

    if metrics["topic_relevance"] >= thresholds["min_topic_relevance"]:
        quality_score += 1
    else:
        reasons.append(f"Low topic relevance ({metrics['topic_relevance']:.2f}, minimum {thresholds['min_topic_relevance']})")

    if metrics["speaker_turns"] >= thresholds["min_speaker_turns"]:
        quality_score += 1
    else:
        reasons.append(f"Too few conversational turns ({metrics['speaker_turns']}, minimum {thresholds['min_speaker_turns']})")

    if metrics["information_density"] >= thresholds["min_information_density"]:
        quality_score += 1
    else:
        reasons.append(f"Low information density ({metrics['information_density']:.2f}, minimum {thresholds['min_information_density']})")

    # 4. To make final quality decision
    quality_ratio = quality_score / max_possible_score if max_possible_score > 0 else 0.0
    is_good_quality = quality_ratio >= 0.7 # at least 70% of criteria be met

    result = {
        "is_good_quality": is_good_quality,
        "quality_score": quality_score,
        "quality_ratio": quality_ratio,
        "metrics": metrics,
        "rejection_reasons": reasons if not is_good_quality else []
    }

    return result
