"""
Test script for Lab 3 audio pipeline.

This script verifies that your environment is correctly configured for the lab.
It checks that the required Python packages are installed, that Google Cloud
credentials are available, and that the Speech and Text-to-Speech APIs can be accessed.
Run this script before starting your homework to catch setup issues early.
"""

import importlib
import sys

REQUIRED_PACKAGES = [
    "google.cloud.speech",
    "google.cloud.texttospeech",
    "librosa",
    "pydub",
    "numpy",
    "spacy",
]

def check_packages():
    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    if missing:
        print("❌ Missing packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("Install them with pip install -r requirements.txt")
    else:
        print("✅ All required packages are installed.")

def check_google_credentials():
    try:
        from google.cloud import speech, texttospeech
        # Attempt to create clients which will force ADC lookup
        speech.SpeechClient()
        texttospeech.TextToSpeechClient()
        print("✅ Google Cloud credentials loaded successfully.")
    except Exception as e:
        print("❌ Failed to load Google Cloud credentials.")
        print(str(e))
        print("Run `gcloud auth application-default login` and ensure APIs are enabled.")

def main():
    print("Running setup tests...")
    check_packages()
    check_google_credentials()
    print("Setup test complete.")

if __name__ == "__main__":
    main()
