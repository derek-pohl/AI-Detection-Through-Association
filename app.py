import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

load_dotenv()

app = Flask(__name__)

# Set up rate limiting (e.g., 5 requests per minute per IP)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["5 per minute"]
)

# Configure the Gemini API
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-2.5-flash')

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

        generated_prompt_response = model.generate_content(prompt_generation_prompt)
        generated_prompt = generated_prompt_response.text

        # Step 2: Generate AI text from the generated prompt
        ai_text_response = model.generate_content(generated_prompt)
        ai_text = ai_text_response.text

        # Step 3: Compare the two texts
        comparison_prompt = f"""Text 1

{user_text}

Text 2

{ai_text}

END OF BOTH TEXTS

Based on ONLY THE IDEAS DISCUSSED in Text 1 and Text 2, how similar are the 2 texts? Which ideas were similar? Which ideas did Text 1 use that Text 2 didn't and vice versa? I'm not asking about the core arguments, I'm talking about the supporting details. It's fine if the texts share similar ideas and there are no original ideas in either."""

        comparison_response = model.generate_content(comparison_prompt)
        comparison_result = comparison_response.text

        # Step 4: Format the comparison into JSON
        json_format_prompt = f"""Take the following text and format it into a JSON object. The JSON should have three keys: "similar_ideas", "text1_original_ideas", and "text2_original_ideas". Each key should have an array of strings as its value, where each string is a distinct idea.

Here is the text to format:
{comparison_result}

Only output the raw JSON, nothing else. Do not wrap it in markdown backticks. It's fine if the texts share similar ideas and there are no original ideas."""

        json_response = model.generate_content(json_format_prompt)
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
