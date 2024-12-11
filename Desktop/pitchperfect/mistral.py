from mistralai import Mistral
import json

class MistralAPI:
    def __init__(self, api_key, model="mistral-small-latest"):
        self.model = model
        self.client = Mistral(api_key=api_key)

    def generate_coverLetter(self, cv_text, job_description, additional_thougts = "", model = "mistral-small-latest", **kwargs):

        max_tokens = kwargs.get("max_tokens", 500)
        try:
            # Define the prompt for generating the cover letter
            prompt = f"""
                ### Task:
                Generate a professional and engaging cover letter based on the following CV and job description:
                Show that you alligns with the job description and that you have the skills and experience to succeed in the role and that you share the companies' values.
                Do not invent any skills or experiences that you do not have.
                You must only include text content starting with 'Dear ' and ending with 'Sincerely, NAME'.
                Keep ALL paragraphs concise and to the point to ensure the cover letter is engaging and easy to read and short.
                Your total answer must be between 400 and at most 500 tokens long.
                
                ### Structure:
                - Introduction (1 paragraph)
                    - State clearly in your opening sentence the purpose for your letter and a brief professional introduction.
                    - Specify why you are interested in that specific position and organization.
                    - Provide an overview of the main strengths and skills you will bring to the role.
                    - Show that you share the companies' values (if you know them).
                - Body (2-3 paragraphs)
                    - Cite a couple of examples from your experience that support your ability to be successful in the position or organization.
                    - Try not to simply repeat your resume in paragraph form, complement your resume by offering a little more detail about key experiences.
                    - Discuss what skills you have developed and connect these back to the target role.
                - Conclusion (1 paragraph)
                    - Restate succinctly your interest in the role and why you are a good candidate.
                    - Thank the reader for their time and consideration.

                ### CV:
                {cv_text}
                
                ### Job Description:
                {job_description}
            """
            
            prompt += f"""
                ### Additional Information:
                {additional_thougts}
            """ if len(additional_thougts) else ""
        
            # Prepare the messages for the chat completion
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

            # Send the request to the Mistral API
            chat_response = self.client.chat.complete(
                model=model,
                messages=messages,
                response_format={
                    "type": "text",  # Assuming the response is plain text
                },
                max_tokens=max_tokens,
            )

            return str(chat_response.choices[0].message.content)
        except Exception as e:
            return f"An error occurred while generating the cover letter: {e}"
        
    def get_jobInfos(self, job_description, tires = 2, model = "mistral-small-latest"):
        if tires <= 0:
            return {
                "error": "An error occurred while processing the job description"
            }
        try:
            schema = {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "location": {"type": "string"},
                    "company": {"type": "string"},
                    "requirements": {"type": "array"},
                },
            }
            
            prompt = f"""
                Analyse the provided job description and retrieve the essential informations. You answer should strictly follow this json schema:
                ### Job Description:
                {job_description}
                
                ### JSON Schema:
                {json.dumps(schema)}
                
                ### Answer: 
                Return your answer in short JSON object following the provided JSON Schema.
                """
                
            
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            client = Mistral(api_key=self._api_key)
            chat_response = client.chat.complete(
                model = model,
                messages = messages,
                response_format = {
                    "type": "json_object",
                }
            )
            return json.loads(chat_response.choices[0].message.content)
        except Exception as e:
            return self.get_jobInfos(job_description, tires - 1, model=model)
            
    def get_personInfos(self, cv_text, tries = 2, model = "mistral-small-latest"):
        if tries <= 0:
            return {
                "error": "An error occurred while processing the CV"
            }
        try:
            schema = {
                "type": "object",
                "properties": {
                    "full_name": {"type": "string"},
                    "fist_name" : {"type": "string"},
                    "last_name": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"},
                    "address": {"type": "string"},
                },
            }
            
            prompt = f"""
                Analyse the provided CV and retrieve the essential informations. You answer should strictly follow this json schema:
                ### CV:
                {cv_text}
                
                ### JSON Schema:
                {json.dumps(schema)}
                
                ### Answer: 
                Return your answer in short JSON object following the provided JSON Schema.
                """
                
            
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

            chat_response = self.client.chat.complete(
                model = model,
                messages = messages,
                response_format = {
                    "type": "json_object",
                }
            )
            return json.loads(chat_response.choices[0].message.content)
        except Exception as e:
            return self.get_personInfos(cv_text, tries - 1, model=model)