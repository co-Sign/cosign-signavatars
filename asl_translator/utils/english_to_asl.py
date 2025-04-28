import os
import json
import re
import requests
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

class EnglishToASLConverter:
    """
    Class for converting English text to ASL gloss using language models.
    """
    def __init__(self, api_type="openai"):
        """
        Initialize the converter.
        
        Args:
            api_type (str): The API to use for conversion. Options: "openai", "custom"
        """
        self.api_type = api_type
        
        # Configure API keys
        if api_type == "openai":
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            openai.api_key = self.openai_api_key
    
    def convert(self, english_text):
        """
        Convert English text to ASL gloss.
        
        Args:
            english_text (str): The English text to convert
            
        Returns:
            str: The ASL gloss representation
        """
        if not english_text:
            return ""
        
        if self.api_type == "openai":
            return self._convert_with_openai(english_text)
        elif self.api_type == "custom":
            return self._convert_with_custom_rules(english_text)
        else:
            raise ValueError(f"Unsupported API type: {self.api_type}")
    
    def _convert_with_openai(self, english_text):
        """
        Convert English text to ASL gloss using OpenAI's language model.
        
        Args:
            english_text (str): The English text to convert
            
        Returns:
            str: The ASL gloss representation
        """
        try:
            # Define the prompt for conversion
            prompt = f"""
            Convert the following English text to American Sign Language (ASL) gloss.
            
            ASL gloss rules:
            1. Change to Object-Subject-Verb order when appropriate
            2. Remove articles (a, an, the)
            3. Replace pronouns (I → ME, you → YOU, etc.)
            4. Remove "to be" verbs (is, are, am, was, were)
            5. Uppercase all words
            6. Keep only essential words that convey meaning
            
            English text: {english_text}
            
            ASL gloss:
            """
            
            # Call the OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a specialist in American Sign Language gloss translation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            # Extract and clean the ASL gloss
            asl_gloss = response.choices[0].message.content.strip()
            
            # Remove "ASL gloss:" if it appears in the response
            asl_gloss = re.sub(r'^ASL gloss:\s*', '', asl_gloss, flags=re.IGNORECASE)
            
            return asl_gloss
        
        except Exception as e:
            print(f"Error during OpenAI conversion: {e}")
            # Fall back to custom rules
            return self._convert_with_custom_rules(english_text)
    
    def _convert_with_custom_rules(self, english_text):
        """
        Convert English text to ASL gloss using custom rules.
        
        Args:
            english_text (str): The English text to convert
            
        Returns:
            str: The ASL gloss representation
        """
        # Normalize text
        text = english_text.strip().upper()
        
        # Remove articles
        text = re.sub(r'\b(A|AN|THE)\b', '', text)
        
        # Replace pronouns
        pronoun_map = {
            r'\bI\b': 'ME',
            r'\bMY\b': 'MY',
            r'\bMYSELF\b': 'MYSELF',
            r'\bYOU\b': 'YOU',
            r'\bYOUR\b': 'YOUR',
            r'\bYOURSELF\b': 'YOURSELF',
            r'\bHE\b': 'HE',
            r'\bHIM\b': 'HIM',
            r'\bHIS\b': 'HIS',
            r'\bHIMSELF\b': 'HIMSELF',
            r'\bSHE\b': 'SHE',
            r'\bHER\b': 'HER',
            r'\bHERSELF\b': 'HERSELF',
            r'\bIT\b': 'IT',
            r'\bITS\b': 'ITS',
            r'\bITSELF\b': 'ITSELF',
            r'\bWE\b': 'WE',
            r'\bUS\b': 'US',
            r'\bOUR\b': 'OUR',
            r'\bOURSELVES\b': 'OURSELVES',
            r'\bTHEY\b': 'THEY',
            r'\bTHEM\b': 'THEM',
            r'\bTHEIR\b': 'THEIR',
            r'\bTHEMSELVES\b': 'THEMSELVES'
        }
        
        for pattern, replacement in pronoun_map.items():
            text = re.sub(pattern, replacement, text)
        
        # Remove "to be" verbs
        text = re.sub(r'\b(AM|IS|ARE|WAS|WERE|BE|BEING|BEEN)\b', '', text)
        
        # Remove contractions
        contraction_map = {
            r'\bCAN\'T\b': 'CANNOT',
            r'\bWON\'T\b': 'WILL NOT',
            r'\bDON\'T\b': 'DO NOT',
            r'\bDOESN\'T\b': 'DOES NOT',
            r'\bDIDN\'T\b': 'DID NOT',
            r'\bHAVEN\'T\b': 'HAVE NOT',
            r'\bHASN\'T\b': 'HAS NOT',
            r'\bHADN\'T\b': 'HAD NOT'
        }
        
        for pattern, replacement in contraction_map.items():
            text = re.sub(pattern, replacement, text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Apply basic Object-Subject-Verb transformation
        # This is a simplified approach and doesn't handle complex sentences
        words = text.split()
        if len(words) >= 3 and not text.endswith('?'):
            # Identify potential sentence components
            # This is a very simplified approach
            verb_index = -1
            for i, word in enumerate(words):
                if word.endswith(('ED', 'ING', 'S')) and i > 0:
                    verb_index = i
                    break
            
            if verb_index > 0 and verb_index < len(words) - 1:
                subject = ' '.join(words[:verb_index])
                verb = words[verb_index]
                obj = ' '.join(words[verb_index+1:])
                text = f"{obj} {subject} {verb}"
        
        return text

class BatchEnglishToASLConverter:
    """
    Class for batch converting English texts to ASL gloss.
    """
    def __init__(self, api_type="openai"):
        """
        Initialize the batch converter.
        
        Args:
            api_type (str): The API to use for conversion
        """
        self.converter = EnglishToASLConverter(api_type)
    
    def convert_from_file(self, input_file, output_file):
        """
        Convert English texts from a file to ASL gloss and save the results.
        
        Args:
            input_file (str): Path to the input file (JSON or text)
            output_file (str): Path to save the output (JSON)
        """
        if input_file.endswith('.json'):
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            # Process based on the structure
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str):
                        data[key] = self.converter.convert(value)
                    elif isinstance(value, dict) and 'text' in value:
                        data[key]['gloss'] = self.converter.convert(value['text'])
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, str):
                        data[i] = self.converter.convert(item)
                    elif isinstance(item, dict) and 'text' in item:
                        data[i]['gloss'] = self.converter.convert(item['text'])
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        else:  # Assume text file
            with open(input_file, 'r') as f:
                lines = f.readlines()
            
            results = []
            for line in lines:
                english_text = line.strip()
                if english_text:
                    asl_gloss = self.converter.convert(english_text)
                    results.append({
                        'english': english_text,
                        'asl_gloss': asl_gloss
                    })
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert English text to ASL gloss")
    parser.add_argument("--text", type=str, help="English text to convert")
    parser.add_argument("--input_file", type=str, help="Path to input file (JSON or text)")
    parser.add_argument("--output_file", type=str, help="Path to output file (JSON)")
    parser.add_argument("--api", type=str, default="openai", choices=["openai", "custom"], help="API to use for conversion")
    
    args = parser.parse_args()
    
    if args.text:
        converter = EnglishToASLConverter(api_type=args.api)
        asl_gloss = converter.convert(args.text)
        print(f"English: {args.text}")
        print(f"ASL Gloss: {asl_gloss}")
    
    elif args.input_file and args.output_file:
        batch_converter = BatchEnglishToASLConverter(api_type=args.api)
        batch_converter.convert_from_file(args.input_file, args.output_file)
        print(f"Converted texts from {args.input_file} and saved to {args.output_file}")
    
    else:
        parser.print_help() 