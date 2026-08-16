"""
Fixed Agent - Accurate confusion analysis with visual grounding
IMPROVEMENTS:
- Specific visual instructions
- Structured observation framework
- Prevents hallucinations
- Frame-by-frame guidance
"""

import json
from google import genai
from google.genai import types
import time

from config import Config


def create_segment_analyzer_agent():
    """Create agent - returns model name"""
    return "gemini-2.5-flash"


def analyze_confusion_segment(segment_context: dict, agent) -> dict:
    """
    Send segment to Gemini for detailed visual analysis
    
    Returns:
        {
            "confusion_type": str,
            "same_person": bool,
            "corrected_times": {person_id: seconds},
            "visual_evidence": str,
            "confidence": float
        }
    """
    
    print(f"\n🤖 AI Agent Analysis Starting...")
    print(f"   Type: {segment_context['confusion_type']}")
    print(f"   IDs: {segment_context['ids_involved']}")
    print(f"   Segment: {segment_context['segment_path']}")
    
    # Initialize client
    client = genai.Client(api_key=Config.GOOGLE_API_KEY)
    
    # Upload video
    print(f"   📤 Uploading video...")
    video_file = client.files.upload(file=segment_context['segment_path'])
    
    # Wait for processing with progress (bounded so a stuck upload can't hang forever)
    print(f"   ⏳ Processing video...", end="", flush=True)
    max_wait_seconds = 120
    waited = 0
    while video_file.state == "PROCESSING":
        if waited >= max_wait_seconds:
            raise TimeoutError(
                f"Gemini file processing did not finish within {max_wait_seconds}s"
            )
        print(".", end="", flush=True)
        time.sleep(1)
        waited += 1
        video_file = client.files.get(name=video_file.name)
    print(" ✓")

    if video_file.state == "FAILED":
        raise ValueError(f"Video processing failed")
    
    # Build context-specific prompt
    prompt = build_analysis_prompt(segment_context)
    
    print(f"   🔍 Analyzing video...")
    
    # Call Gemini with strict schema
    response = client.models.generate_content(
        model=agent,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part(text=prompt),
                    types.Part(file_data=types.FileData(file_uri=video_file.uri))
                ]
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,  # Low temperature for accuracy
            response_mime_type="application/json"
        )
    )
    
    result = json.loads(response.text)
    
    # Print analysis summary
    print(f"\n📊 Analysis Results:")
    print(f"   Same Person: {result.get('same_person', 'unknown')}")
    print(f"   Confidence: {result.get('confidence', 0):.0%}")
    print(f"   Evidence: {result.get('visual_evidence', 'N/A')[:150]}...")
    
    if 'corrected_times' in result:
        print(f"\n   ⏱️  Corrected Times:")
        for pid, time_val in result['corrected_times'].items():
            if time_val is not None:
                try:
                    time_val_float = float(time_val)
                    print(f"      ID {pid}: {time_val_float:.1f}s ({time_val_float/60:.1f} min)")
                except (ValueError, TypeError):
                    print(f"      ID {pid}: {time_val} (invalid format)")
            else:
                print(f"      ID {pid}: None (no correction provided)")
    
    # Cleanup uploaded file
    try:
        client.files.delete(name=video_file.name)
        print(f"   🗑️  Cleaned up uploaded file")
    except Exception as e:
        print(f"   ⚠️  Cleanup warning: {e}")
    
    return result


def build_analysis_prompt(segment_context: dict) -> str:
    """
    Build confusion-type specific prompt with visual grounding
    """
    
    confusion_type = segment_context['confusion_type']
    ids = segment_context['ids_involved']
    time_data = segment_context['id_time_data']
    
    # Base context
    base_prompt = f"""You are analyzing a video segment from a retail store's security camera.

CONFUSION TYPE: {confusion_type}
IDs INVOLVED: {ids}

TRACKING DATA BEFORE CONFUSION:
"""
    
    for pid, data in time_data.items():
        base_prompt += f"\nID {pid}:"
        base_prompt += f"\n  - First seen: {data['first_seen']:.1f}s"
        base_prompt += f"\n  - Last seen before confusion: {data['last_seen']:.1f}s"
        base_prompt += f"\n  - Time tracked: {data['time_before_confusion']:.1f}s"
    
    # Type-specific instructions
    if confusion_type == 'id_switch':
        context = segment_context['confusion_details']
        old_id = context['old_id']
        new_id = context['new_id']
        
        specific_prompt = f"""

CONFUSION DETAILS:
- Person was tracked as ID {old_id}
- Then suddenly became ID {new_id}
- Distance between last/first detection: {context.get('distance', 'unknown')} pixels

YOUR TASK:
Watch the video carefully and answer:

1. VISUAL OBSERVATION:
   - Describe the person's appearance (clothing color, size, distinctive features)
   - Track their movement through the frames
   - Note any moment where tracking seems to fail or jump

2. IDENTITY DETERMINATION:
   - Is this ONE person who was re-identified with a new ID?
   - Or are these TWO different people?
   
3. EVIDENCE:
   Look for:
   - Clothing continuity (same shirt/pants colors?)
   - Body size/proportions consistency
   - Movement patterns (walking speed, direction)
   - Any occlusion or camera angle changes that could cause re-identification

4. TIME CORRECTION:
   If SAME PERSON (ID switch):
   - Merge their times: ID {old_id}'s time + ID {new_id}'s time
   - Report corrected time under ID {new_id} (the final ID)
   
   If DIFFERENT PEOPLE:
   - Keep times separate
   - Report both IDs with their original times

Return JSON:
{{
  "confusion_type": "id_switch",
  "same_person": true or false,
  "visual_evidence": "Detailed description of what you observed - clothing, movement, any occlusions",
  "corrected_times": {{
    "{new_id}": <total_seconds_if_merged>,
    "{old_id}": <original_seconds_if_separate>
  }},
  "confidence": 0.0 to 1.0
}}

CRITICAL: Base your answer ONLY on what you see in the video. Do not guess or extrapolate."""

    elif confusion_type == 'occlusion':
        pid = ids[0]
        specific_prompt = f"""

CONFUSION DETAILS:
- Person ID {pid} disappeared briefly from tracking
- They reappeared after a short time

YOUR TASK:
Watch the video and determine:

1. VISUAL OBSERVATION:
   - When does person ID {pid} disappear from view?
   - WHY did they disappear? (walked behind shelf, blocked by another person, left frame, etc.)
   - Do they reappear in the video?

2. CONTINUITY CHECK:
   - Is the person who reappears the SAME person who disappeared?
   - Check: same clothing, same body type, similar location
   
3. TIME CORRECTION:
   If SAME PERSON (true occlusion):
   - Keep their time continuous (no time deduction for occlusion)
   - Report total time from first appearance to last appearance
   
   If DIFFERENT PERSON (false detection):
   - This shouldn't happen for occlusion type, but report if you see it
   - Keep times separate

Return JSON:
{{
  "confusion_type": "occlusion",
  "same_person": true or false,
  "visual_evidence": "What caused the occlusion? What did you see?",
  "corrected_times": {{
    "{pid}": <total_seconds_from_first_to_last_appearance>
  }},
  "confidence": 0.0 to 1.0
}}

CRITICAL: Describe exactly what you see causing the occlusion."""

    elif confusion_type == 'return_after_leave':
        pid = ids[0]
        time_away = segment_context['confusion_details'].get('time_away', 'unknown')
        
        specific_prompt = f"""

CONFUSION DETAILS:
- Person ID {pid} left the tracked area
- They returned after {time_away}s

YOUR TASK:
This is tricky - determine if this is truly the SAME person returning, or a NEW person who got assigned the same ID.

1. VISUAL COMPARISON:
   - Compare the person at the START of the video (before leaving)
   - To the person at the END of the video (after returning)
   - Are they wearing the same clothes?
   - Same body type and size?
   - Same movement patterns?

2. BEHAVIORAL CLUES:
   - Did they exit through a door/exit area?
   - Did they return through the same entrance?
   - Or did they just move to a different part of the store?

3. TIME CORRECTION:
   If SAME PERSON (true return):
   - This is a legitimate customer who left and came back
   - Count BOTH visit times separately OR merged (your decision based on store policy)
   - Recommended: Keep separate visits as separate entries
   
   If DIFFERENT PERSON (ID reuse):
   - The system reused the ID for a new customer
   - Keep times separate
   - Original person: time until they left
   - New person: time from when they "returned"

Return JSON:
{{
  "confusion_type": "return_after_leave",
  "same_person": true or false,
  "visual_evidence": "Detailed comparison of clothing, body type, behavior",
  "corrected_times": {{
    "{pid}_visit1": <seconds_for_first_visit>,
    "{pid}_visit2": <seconds_for_second_visit_if_same_person>,
    "{pid}": <total_seconds_if_merged_or_if_different_person>
  }},
  "confidence": 0.0 to 1.0
}}

CRITICAL: Pay close attention to clothing and body type. If they're different, it's definitely a different person."""

    else:
        specific_prompt = f"""

YOUR TASK:
Analyze this confusion event and provide visual evidence.

Return JSON with your observations."""

    return base_prompt + specific_prompt


if __name__ == "__main__":
    # Test agent creation
    agent = create_segment_analyzer_agent()
    print(f"✅ Agent model: {agent}")
    
    # Test with mock data
    mock_segment = {
        'confusion_type': 'id_switch',
        'segment_path': 'test.mp4',
        'ids_involved': [45, 46],
        'id_time_data': {
            45: {'first_seen': 10.0, 'last_seen': 50.0, 'time_before_confusion': 40.0},
            46: {'first_seen': 50.5, 'last_seen': 80.0, 'time_before_confusion': 29.5}
        },
        'confusion_details': {
            'old_id': 45,
            'new_id': 46,
            'distance': 150.2
        }
    }
    
    print("\n📋 Example prompt structure:")
    print("="*60)
    print(build_analysis_prompt(mock_segment)[:500] + "...")