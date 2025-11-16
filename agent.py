"""
Simplified ADK Agent - Watches segment and fixes time calculation
"""

import os
import json
from google.adk.agents import LlmAgent
from google import genai


def create_segment_analyzer_agent():
    """
    Create ADK agent that analyzes confusion segments using vision
    """
    
    agent = LlmAgent(
        name="confusion_segment_analyzer",
        model="gemini-2.5-flash",  # Has vision capabilities
        description="Analyzes confusion segments and corrects customer time calculations",
        
        instruction="""
You are analyzing a video segment where customer tracking got confused.

YOUR TASK:
1. Watch the provided video segment carefully
2. You will receive:
   - Video segment (5-10 seconds around confusion point)
   - Confusion type (id_switch, occlusion, or return_after_leave)
   - IDs involved
   - Time each person spent BEFORE the confusion happened
3. Analyze the segment and determine:
   - Are the IDs the same person or different people?
   - Should their times be merged or kept separate?
   - What is the corrected total time for each person?

OUTPUT FORMAT (JSON):
{
  "confusion_type": "...",
  "decision": "merge" or "separate",
  "reasoning": "detailed explanation of what you saw",
  "corrected_times": {
    "person_id": corrected_time_in_seconds,
    ...
  },
  "confidence": 0.0 to 1.0
}

Be specific about what you observe in the video that led to your decision.
        """
    )
    
    return agent


def analyze_confusion_segment(segment_context: dict, agent: LlmAgent) -> dict:
    """
    Send segment to agent for analysis
    
    Args:
        segment_context: Context from segment extractor
        agent: ADK agent instance
    
    Returns:
        Corrected time calculation from agent
    """
    
    print(f"\n🤖 Analyzing segment with AI agent...")
    print(f"   Type: {segment_context['confusion_type']}")
    print(f"   IDs: {segment_context['ids_involved']}")
    
    # Upload video segment to Gemini
    client = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))
    
    video_file = client.files.upload(path=segment_context['segment_path'])
    
    # Wait for processing
    import time
    while video_file.state == "PROCESSING":
        time.sleep(1)
        video_file = client.files.get(name=video_file.name)
    
    print(f"   Video uploaded: {video_file.name}")
    
    # Build prompt with context
    prompt = f"""
Analyze this confusion segment:

CONFUSION TYPE: {segment_context['confusion_type']}
IDs INVOLVED: {segment_context['ids_involved']}

TIME DATA BEFORE CONFUSION:
{json.dumps(segment_context['id_time_data'], indent=2)}

CONFUSION DETAILS:
{json.dumps(segment_context['confusion_details'], indent=2)}

Watch the video carefully and determine:
1. Are these IDs the same person or different people?
2. Should their times be merged or kept separate?
3. What are the corrected total times?

Return your analysis in JSON format.
    """
    
    # Send to agent with video
    from google.genai import types
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
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
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "confusion_type": {"type": "string"},
                    "decision": {"type": "string", "enum": ["merge", "separate"]},
                    "reasoning": {"type": "string"},
                    "corrected_times": {
                        "type": "object",
                        "additionalProperties": {"type": "number"}
                    },
                    "confidence": {"type": "number"}
                }
            }
        )
    )
    
    result = json.loads(response.text)
    
    print(f"✅ Agent decision: {result['decision'].upper()}")
    print(f"   Confidence: {result['confidence']:.0%}")
    print(f"   Reasoning: {result['reasoning'][:100]}...")
    
    # Cleanup
    client.files.delete(name=video_file.name)
    
    return result


if __name__ == "__main__":
    agent = create_segment_analyzer_agent()
    print(f"✅ Agent created: {agent.name}")
