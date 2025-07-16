import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import threading
import concurrent.futures
import json
from collections import defaultdict

load_dotenv()

app = Flask(__name__)

# Set up rate limiting (e.g., 5 requests per minute per IP)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["5 per minute"]
)

# API Key rotation setup
class APIKeyRotator:
    def __init__(self):
        # Parse comma-separated API keys from environment
        api_keys_str = os.environ.get("GOOGLE_API_KEYS", os.environ.get("GOOGLE_API_KEY", ""))
        self.api_keys = [key.strip() for key in api_keys_str.split(",") if key.strip()]
        
        if not self.api_keys:
            raise ValueError("No API keys found. Please set GOOGLE_API_KEYS or GOOGLE_API_KEY in your .env file")
        
        self.current_index = 0
        self.lock = threading.Lock()
        
        print(f"Initialized with {len(self.api_keys)} API key(s)")
    
    def get_next_key(self):
        with self.lock:
            key = self.api_keys[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            return key
    
    def get_model_with_next_key(self):
        api_key = self.get_next_key()
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.5-flash')

# Initialize the API key rotator
key_rotator = APIKeyRotator()

def run_single_cycle(user_text, cycle_id):
    """Run a single analysis cycle and return the results"""
    try:
        # Step 1: Generate a prompt for the AI
        prompt_generation_prompt = f"""{user_text}

The text above was written by Generative AI. Based on the features of the text, eg, how many paragraphs it is, or the BROAD topic of the text, write a prompt to an AI that would output a similar answer. Let the AI determine what examples to use to answer the question, DO NOT guide the AI in the prompt on what specific examples to use.

Good Prompt: In a 200 word paragraph, how did people respond to the new economic and social structure created by the Industrial Revolution?

Bad Prompt: Write a concise historical overview detailing how people responded to the significant economic and social changes brought about by the Industrial Revolution. Focus on the range of reactions, including resistance to new work structures, attempts at adaptation, and the emergence of more radical demands for change. Discuss how the shift from traditional labor systems impacted workers and contributed to social and political movements aimed at improving working conditions.

The BAD prompt was bad due to it being very specific.

Only output the prompt, nothing else."""

        # Step 1: Use rotated API key for prompt generation
        model1 = key_rotator.get_model_with_next_key()
        generated_prompt_response = model1.generate_content(prompt_generation_prompt)
        generated_prompt = generated_prompt_response.text

        # Step 2: Generate AI text from the generated prompt (use next key in rotation)
        model2 = key_rotator.get_model_with_next_key()
        ai_text_response = model2.generate_content(generated_prompt)
        ai_text = ai_text_response.text

        # Step 3: Compare the two texts (use next key in rotation)
        comparison_prompt = f"""Text 1

{user_text}

Text 2

{ai_text}

END OF BOTH TEXTS

Based on ONLY THE IDEAS DISCUSSED in Text 1 and Text 2, how similar are the 2 texts? Which ideas were similar? Which ideas did Text 1 use that Text 2 didn't and vice versa? I'm not asking about the core arguments, I'm talking about the supporting details. It's fine if the texts share similar ideas and there are no original ideas in either."""

        model3 = key_rotator.get_model_with_next_key()
        comparison_response = model3.generate_content(comparison_prompt)
        comparison_result = comparison_response.text

        # Step 4: Format the comparison into JSON (use next key in rotation)
        json_format_prompt = f"""Take the following text and format it into a JSON object. The JSON should have three keys: "similar_ideas", "text1_original_ideas", and "text2_original_ideas". Each key should have an array of strings as its value, where each string is a distinct idea.

Here is the text to format:
{comparison_result}

Only output the raw JSON, nothing else. Do not wrap it in markdown backticks. It's fine if the texts share similar ideas and there are no original ideas."""

        model4 = key_rotator.get_model_with_next_key()
        json_response = model4.generate_content(json_format_prompt)
        # Clean up the response to ensure it's valid JSON
        json_result_text = json_response.text.strip().replace('```json', '').replace('```', '')

        cycle_json = json.loads(json_result_text)

        return {
            'cycle_id': cycle_id,
            'success': True,
            'result': cycle_json,
            'details': {
                'generated_prompt': generated_prompt,
                'ai_text': ai_text,
                'comparison_result': comparison_result
            }
        }
    except Exception as e:
        return {
            'cycle_id': cycle_id,
            'success': False,
            'error': str(e)
        }

def average_results(cycle_results):
    """Average the results from multiple cycles"""
    successful_cycles = [cycle for cycle in cycle_results if cycle['success']]
    
    if not successful_cycles:
        raise Exception("No successful cycles completed")
    
    # Collect all ideas from all cycles
    all_similar_ideas = []
    all_text1_original_ideas = []
    all_text2_original_ideas = []
    
    for cycle in successful_cycles:
        result = cycle['result']
        all_similar_ideas.extend(result.get('similar_ideas', []))
        all_text1_original_ideas.extend(result.get('text1_original_ideas', []))
        all_text2_original_ideas.extend(result.get('text2_original_ideas', []))
    
    # Count frequency of each idea
    similar_idea_counts = defaultdict(int)
    text1_idea_counts = defaultdict(int)
    text2_idea_counts = defaultdict(int)
    
    for idea in all_similar_ideas:
        similar_idea_counts[idea] += 1
    for idea in all_text1_original_ideas:
        text1_idea_counts[idea] += 1
    for idea in all_text2_original_ideas:
        text2_idea_counts[idea] += 1
    
    # Calculate average scores (ideas that appear in majority of cycles)
    num_cycles = len(successful_cycles)
    threshold = num_cycles / 2  # Ideas that appear in more than half the cycles
    
    averaged_similar_ideas = [idea for idea, count in similar_idea_counts.items() if count > threshold]
    averaged_text1_ideas = [idea for idea, count in text1_idea_counts.items() if count > threshold]
    averaged_text2_ideas = [idea for idea, count in text2_idea_counts.items() if count > threshold]
    
    return {
        'similar_ideas': averaged_similar_ideas,
        'text1_original_ideas': averaged_text1_ideas,
        'text2_original_ideas': averaged_text2_ideas,
        'cycle_stats': {
            'total_cycles': len(cycle_results),
            'successful_cycles': len(successful_cycles),
            'failed_cycles': len(cycle_results) - len(successful_cycles)
        }
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/how-it-works')
def how_it_works():
    return render_template('how-it-works.html')

@app.route('/analyze', methods=['POST'])
@limiter.limit("5 per minute")  # Optional: override or reinforce per-endpoint limit
def analyze():
    try:
        user_text = request.json['text']

        # Step 1: Generate a prompt for the AI
        prompt_generation_prompt = f"""{user_text}

The text above was written by Generative AI. Based on the features of the text, eg, how many paragraphs it is, or the BROAD topic of the text, write a prompt to an AI that would output a similar answer. Let the AI determine what examples to use to answer the question, DO NOT guide the AI in the prompt on what specific examples to use.

Good Prompt: In a 200 word paragraph, how did people respond to the new economic and social structure created by the Industrial Revolution?

Bad Prompt: Write a concise historical overview detailing how people responded to the significant economic and social changes brought about by the Industrial Revolution. Focus on the range of reactions, including resistance to new work structures, attempts at adaptation, and the emergence of more radical demands for change. Discuss how the shift from traditional labor systems impacted workers and contributed to social and political movements aimed at improving working conditions.

The BAD prompt was bad due to it being very specific.

Only output the prompt, nothing else."""

        # Step 1: Use rotated API key for prompt generation
        model1 = key_rotator.get_model_with_next_key()
        generated_prompt_response = model1.generate_content(prompt_generation_prompt)
        generated_prompt = generated_prompt_response.text

        # Step 2: Generate AI text from the generated prompt (use next key in rotation)
        model2 = key_rotator.get_model_with_next_key()
        ai_text_response = model2.generate_content(generated_prompt)
        ai_text = ai_text_response.text

        # Step 3: Compare the two texts (use next key in rotation)
        comparison_prompt = f"""Text 1

{user_text}

Text 2

{ai_text}

END OF BOTH TEXTS

Based on ONLY THE IDEAS DISCUSSED in Text 1 and Text 2, how similar are the 2 texts? Which ideas were similar? Which ideas did Text 1 use that Text 2 didn't and vice versa? I'm not asking about the core arguments, I'm talking about the supporting details. It's fine if the texts share similar ideas and there are no original ideas in either."""

        model3 = key_rotator.get_model_with_next_key()
        comparison_response = model3.generate_content(comparison_prompt)
        comparison_result = comparison_response.text

        # Step 4: Format the comparison into JSON (use next key in rotation)
        json_format_prompt = f"""Take the following text and format it into a JSON object. The JSON should have three keys: "similar_ideas", "text1_original_ideas", and "text2_original_ideas". Each key should have an array of strings as its value, where each string is a distinct idea.

Here is the text to format:
{comparison_result}

Only output the raw JSON, nothing else. Do not wrap it in markdown backticks. It's fine if the texts share similar ideas and there are no original ideas."""

        model4 = key_rotator.get_model_with_next_key()
        json_response = model4.generate_content(json_format_prompt)
        # Clean up the response to ensure it's valid JSON
        json_result_text = json_response.text.strip().replace('```json', '').replace('```', '')

        import json
        final_json = json.loads(json_result_text)

        return jsonify({
            'final_result': final_json,
            'details': {
                'generated_prompt': generated_prompt,
                'ai_text': ai_text,
                'comparison_result': comparison_result
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
else:
    # For Vercel deployment
    pass
