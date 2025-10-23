# Google Cloud Setup Guide



This guide walks you through configuring your Google Cloud environment for Lab 3.  You will need a Google account with billing enabled and access to the Speech‑to‑Text and Text‑to‑Speech APIs.



## 1. Install the Cloud SDK



Download and install the [Google Cloud SDK](https://cloud.google.com/sdk) for your operating system.  Follow the official installation instructions.  Once installed, open a terminal and run:



```bash

gcloud init

```



Choose an existing Google Cloud project or create a new one.  The project will be used for all API calls.



## 2. Authenticate and set application default credentials



The Python client libraries use Application Default Credentials to access Google services.  To generate these credentials run:



```bash

gcloud auth application-default login

```



Sign in with your Google account when prompted.  A local credentials file will be created and used by the client libraries.



## 3. Enable the APIs



Enable the Speech‑to‑Text and Text‑to‑Speech services for your project:



```bash

gcloud services enable speech.googleapis.com

gcloud services enable texttospeech.googleapis.com

```



These commands may take a few seconds to complete.  You only need to run them once per project.



## 4. Billing setup



Google Cloud’s free tier includes 60 minutes of Speech‑to‑Text and 4 million characters of Text‑to‑Speech output per month.  You must still link a billing account to your project, even if you remain within the free tier.  Visit the [Billing section](https://console.cloud.google.com/billing) of the Cloud Console and link a payment method if you have not done so already.



## 5. Install Python dependencies



Create a virtual environment and install the requirements listed in `starter_code/requirements.txt`:



```bash

python3 -m venv venv

source venv/bin/activate

pip install -r starter_code/requirements.txt

```



This will install `google‑cloud‑speech`, `google‑cloud‑texttospeech`, `librosa`, `numpy`, `pydub`, `spacy` and their dependencies.



## 6. Test your setup



Run the provided test script to ensure your environment is configured correctly:



```bash

python scripts/test_setup.py

```



The script checks for:



- A valid Application Default Credentials file.

- Enabled Speech‑to‑Text and Text‑to‑Speech APIs.

- Installed Python packages.



Fix any reported issues before starting the lab.



## 7. Troubleshooting



- **Application Default Credentials not found** – run `gcloud auth application-default login` again.

- **API not enabled** – enable the service using `gcloud services enable speech.googleapis.com` or `texttospeech.googleapis.com`.

- **Billing account not configured** – link a billing account under the Billing section of the Cloud Console.

- **ModuleNotFoundError** – install missing packages with `pip install -r starter_code/requirements.txt`.



If you continue to encounter issues, ask for help in the course forum or during office hours.
