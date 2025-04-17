import os
from src.quality_assessment import assess_call_quality
from src.proficiency_assessment import assess_english_level
from notebooks.proficiency_visualization import generate_proficiency_graph

def process_transcript(transcript_text: str):
    """
    Processes a single transcript using the assessment modules.

    Args:
        transcript_text: The string content of the transcript.
    """
    print("-" * 30)
    print("Processing Transcript:")
    # the preview of the transcript
    print(f"\"{transcript_text[:100]}...\"")
    print("-" * 30)

    # 1. To assess quality
    print("\nAssessing Quality...")
    quality_result = assess_call_quality(transcript_text)
    if quality_result:
        print(f"  Good Quality: {quality_result.get('is_good_quality')}")
        print(f"  Quality Score: {quality_result.get('quality_score')}/{len(quality_result.get('metrics', {}))}")
        print(f"  Quality Ratio: {quality_result.get('quality_ratio'):.2f}")
        if not quality_result.get('is_good_quality'):
            print(f"  Rejection Reasons: {quality_result.get('rejection_reasons')}")
        # to print detailed metrics
         print("  Detailed Metrics:", quality_result.get('metrics'))
    else:
        print("  Could not assess quality.")


    # 2. to assess English Proficiency
    print("\nAssessing English Proficiency...")
    proficiency_level = assess_english_level(transcript_text)
    print(f"  Estimated CEFR Level: {proficiency_level}")

    print("-" * 30)

# ---The Main Execution ---
if __name__ == "__main__":
    # --- Option 1: Use my sample transcript here to test ---
    sample_transcript = (
        "Agent: Thank you for calling Tech Support, this is Alex speaking. How may I help you today? "
        "User: Hi Alex, uhm, yes, my computer... it is not starting. It makes some noise, like beeps? "
        "Agent: Okay, I understand. Can you describe the beeps? Are they long or short? How many? "
        "User: Uh, maybe three short beeps? Yes, three shorts. Then nothing. Screen is black. "
        "Agent: Three short beeps often indicate a memory issue. Did anything change recently? Did you install new RAM? "
        "User: No, I not change anything. It just... stop working this morning. Very bad. I need computer for work. "
        "Agent: I see. Let's try a basic troubleshooting step. Could you try reseating the memory modules? "
        "User: Resetting? How I do this? Is it difficult?"
    )
    process_transcript(sample_transcript)

    # --- Option 2: Load transcript from your file (Example 'uncomment below code') ---
    # transcript_file = "path/to/your/transcript.txt"
    # try:
    #     with open(transcript_file, 'r', encoding='utf-8') as f:
    #         file_transcript = f.read()
    #     process_transcript(file_transcript)
    # except FileNotFoundError:
    #     print(f"\nError: Transcript file not found at {transcript_file}")
    # except Exception as e:
    #     print(f"\nError reading transcript file: {e}")

    print("\nScript finished.")

