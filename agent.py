"""
Simplified Agent - Watches segment and fixes time calculation
FIXED: Gemini schema format, file upload API
"""

import os
import json
from google import genai
from google.genai import types


def create_segment_analyzer_agent():
    """Create agent - returns model name"""
    return "gemini-2.5-flash"


def analyze_confusion_segment(segment_context: dict, agent) -> dict:
    """
    Send segment to Gemini for analysis
    """
    
    print(f"\n🤖 Analyzing segment with AI agent...")
    print(f"   Type: {segment_context['confusion_type']}")
    print(f"   IDs: {segment_context['ids_involved']}")
    
    # Initialize client
    client = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))
    
    # Upload video
    print(f"   Uploading video: {segment_context['segment_path']}")
    video_file = client.files.upload(file=segment_context['segment_path'])
    
    # Wait for processing
    import time
    while video_file.state == "PROCESSING":
        print("   Processing video...")
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)
    
    if video_file.state == "FAILED":
        raise ValueError(f"Video processing failed")
    
    print(f"   Video ready: {video_file.name}")
    
    # Build prompt
    prompt = f"""
Analyze this confusion segment:

CONFUSION TYPE: {segment_context['confusion_type']}
IDs INVOLVED: {segment_context['ids_involved']}

TIME DATA BEFORE CONFUSION:
{json.dumps(segment_context['id_time_data'], indent=2)}

Watch the video and determine:
1. Are these IDs the same person or different people?
2. Should their times be merged or kept separate?
3. What are the corrected total times for each person ID?

Return JSON with:
- confusion_type: string
- decision: "merge" or "separate"
- reasoning: string explaining what you saw
- corrected_times: dict where keys are person IDs (as strings) and values are corrected times in seconds
- confidence: number 0.0 to 1.0

Example output:
{{
  "confusion_type": "occlusion",
  "decision": "separate",
  "reasoning": "Video shows...",
  "corrected_times": {{"60": 120.5}},
  "confidence": 0.85
}}
"""
    
    # ✅ FIXED: Simplified schema without additionalProperties
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
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
            temperature=0.1,
            response_mime_type="application/json"
        )
    )
    
    result = json.loads(response.text)
    
    print(f"✅ Agent decision: {result.get('decision', 'unknown').upper()}")
    print(f"   Confidence: {result.get('confidence', 0):.0%}")
    print(f"   Reasoning: {result.get('reasoning', 'N/A')[:100]}...")
    
    # Cleanup
    try:
        client.files.delete(name=video_file.name)
    except:
        pass
    
    return result


if __name__ == "__main__":
    agent = create_segment_analyzer_agent()
    print(f"✅ Agent model: {agent}")
