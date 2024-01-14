### Chapter: Setting Up and Running `vison_llm_gemini_voice_plus(_en).py`

This section provides a comprehensive guide on preparing and executing the `vison_llm_gemini_voice_plus.py` script. It's essential to configure specific environment variables and install various Python packages before running the script.

#### Setting Environment Variables

To ensure the script functions correctly, set the following environment variables:

1. Setting `PICOVOICE_KEYWORD_PATH`:
   ```bash
   export PICOVOICE_KEYWORD_PATH=./Hey-Gemini_en_mac_v3_0_0.ppn
   ```
   For more information on Picovoice keywords, visit the [Picovoice Python API documentation](https://picovoice.ai/docs/api/porcupine-python/).

2. Setting `PICOVOICE_ACCESS_KEY`:
   ```bash
   export PICOVOICE_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

3. Setting `GOOGLE_API_KEY`:
   ```bash
   export GOOGLE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
   For details on obtaining a Google API key, refer to the [Google Maker Suite documentation](https://makersuite.google.com/app/apikey).

#### Installing Python Packages

The following Python packages are required for the script. Install them using these commands:

1. `pvporcupine`:
   ```bash
   pip install pvporcupine
   ```

2. Google Cloud libraries:
   ```bash
   pip install google-cloud-speech google-cloud-texttospeech
   ```

3. `pyaudio`:
   ```bash
   pip install pyaudio
   ```

4. OpenCV:
   ```bash
   pip install opencv-python
   ```

5. `pydub`:
   ```bash
   pip install pydub
   ```

6. Pillow (PIL):
   ```bash
   pip install Pillow
   ```

7. `google.generativeai` (Note: This package may not be available in the standard Python Package Index):
   ```bash
   pip install google.generativeai
   ```

#### Running the Script

After configuring the environment variables and installing the packages, execute the script with the command below:

```bash
python vison_llm_gemini_voice_plus.py
```