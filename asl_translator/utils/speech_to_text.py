import os
import time
import speech_recognition as sr
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SpeechToText:
    """
    Class for converting speech to text using various speech recognition APIs.
    """
    def __init__(self, api="google", timeout=5):
        """
        Initialize the speech-to-text converter.
        
        Args:
            api (str): The API to use for speech recognition. Options: "google", "sphinx", "wit", "azure", "ibm"
            timeout (int): The maximum time to wait for speech recognition, in seconds
        """
        self.recognizer = sr.Recognizer()
        self.api = api
        self.timeout = timeout
        
        # Configure API keys if needed
        if api == "wit":
            self.wit_api_key = os.getenv("WIT_API_KEY")
            if not self.wit_api_key:
                raise ValueError("WIT_API_KEY environment variable is not set")
        elif api == "azure":
            self.azure_api_key = os.getenv("AZURE_SPEECH_KEY")
            self.azure_region = os.getenv("AZURE_SPEECH_REGION")
            if not self.azure_api_key or not self.azure_region:
                raise ValueError("AZURE_SPEECH_KEY or AZURE_SPEECH_REGION environment variable is not set")
        elif api == "ibm":
            self.ibm_username = os.getenv("IBM_SPEECH_USERNAME")
            self.ibm_password = os.getenv("IBM_SPEECH_PASSWORD")
            if not self.ibm_username or not self.ibm_password:
                raise ValueError("IBM_SPEECH_USERNAME or IBM_SPEECH_PASSWORD environment variable is not set")
    
    def recognize_from_microphone(self):
        """
        Recognize speech from microphone input.
        
        Returns:
            str: The recognized text
        """
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source)
            try:
                audio = self.recognizer.listen(source, timeout=self.timeout)
            except sr.WaitTimeoutError:
                print("No speech detected within timeout period")
                return ""
        
        return self._process_audio(audio)
    
    def recognize_from_file(self, audio_file_path):
        """
        Recognize speech from an audio file.
        
        Args:
            audio_file_path (str): Path to the audio file
            
        Returns:
            str: The recognized text
        """
        with sr.AudioFile(audio_file_path) as source:
            audio = self.recognizer.record(source)
        
        return self._process_audio(audio)
    
    def _process_audio(self, audio):
        """
        Process audio data using the selected API.
        
        Args:
            audio: Audio data from the recognizer
            
        Returns:
            str: The recognized text
        """
        try:
            if self.api == "google":
                return self.recognizer.recognize_google(audio)
            elif self.api == "sphinx":
                return self.recognizer.recognize_sphinx(audio)
            elif self.api == "wit":
                return self.recognizer.recognize_wit(audio, key=self.wit_api_key)
            elif self.api == "azure":
                return self.recognizer.recognize_azure(audio, key=self.azure_api_key, location=self.azure_region)
            elif self.api == "ibm":
                return self.recognizer.recognize_ibm(audio, username=self.ibm_username, password=self.ibm_password)
            else:
                raise ValueError(f"Unsupported API: {self.api}")
        except sr.UnknownValueError:
            print("Speech recognition could not understand audio")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results from service; {e}")
            return ""
        except Exception as e:
            print(f"Error during speech recognition: {e}")
            return ""

class ContinuousSpeechRecognition:
    """
    Class for continuous speech recognition.
    """
    def __init__(self, api="google", pause_threshold=1.0, callback=None):
        """
        Initialize the continuous speech recognition.
        
        Args:
            api (str): The API to use for speech recognition
            pause_threshold (float): The minimum length of silence to consider the end of a phrase
            callback (callable): Function to call with the recognized text
        """
        self.speech_to_text = SpeechToText(api)
        self.recognizer = self.speech_to_text.recognizer
        self.recognizer.pause_threshold = pause_threshold
        self.callback = callback
        self.running = False
    
    def start(self):
        """
        Start continuous speech recognition.
        """
        self.running = True
        
        with sr.Microphone() as source:
            print("Continuous speech recognition started. Speak now...")
            self.recognizer.adjust_for_ambient_noise(source)
            
            while self.running:
                try:
                    audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=5)
                    text = self.speech_to_text._process_audio(audio)
                    
                    if text and self.callback:
                        self.callback(text)
                except sr.WaitTimeoutError:
                    continue
                except KeyboardInterrupt:
                    self.stop()
                except Exception as e:
                    print(f"Error during continuous recognition: {e}")
    
    def stop(self):
        """
        Stop continuous speech recognition.
        """
        self.running = False
        print("Continuous speech recognition stopped.")

if __name__ == "__main__":
    # Example usage
    speech_to_text = SpeechToText()
    
    print("Say something...")
    text = speech_to_text.recognize_from_microphone()
    print(f"Recognized: {text}")
    
    # Example of continuous recognition
    def print_result(text):
        print(f"Recognized: {text}")
    
    continuous_recognition = ContinuousSpeechRecognition(callback=print_result)
    try:
        continuous_recognition.start()
    except KeyboardInterrupt:
        continuous_recognition.stop() 