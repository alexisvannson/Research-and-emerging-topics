from flask import Flask, request, jsonify, send_file
from mistral import MistralAPI
import os
from dotenv import load_dotenv
from fpdf import FPDF
from io import BytesIO

app = Flask(__name__)

# Load environment variables
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Initialize MistralAPI
mistral = MistralAPI(api_key=MISTRAL_API_KEY)

# Route for generating cover letter
@app.route('/generate_cover_letter', methods=['POST'])
def generate_cover_letter():
    cv_file = request.files.get('cv')
    job_description = request.form.get('jobDescription')
    additional_thoughts = request.form.get('additionalThoughts', "")

    if not cv_file or not job_description:
        return jsonify({"error": "CV file and job description are required"}), 400

    # Read CV content
    cv_text = cv_file.read().decode('utf-8')

    # Generate cover letter
    cover_letter = mistral.generate_coverLetter(cv_text, job_description, additional_thoughts)

    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, cover_letter)

    # Return PDF as response
    pdf_bytes = BytesIO()
    pdf.output(pdf_bytes)
    pdf_bytes.seek(0)

    return send_file(pdf_bytes, download_name="cover_letter.pdf", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
