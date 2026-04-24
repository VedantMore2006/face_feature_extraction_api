# API SUBMISSION GUIDELINES
**Development Team - Saraswati College**
*Prepared by: Vicky*

---

## API SUBMISSION REQUIREMENTS
Every API submitted by a developer must include the following:

1. **README File**
   A concise, to-the-point README describing what the API does, its endpoints, and how to run it. It must also include a "How to Use" section showing how to call the API using Python's requests library:

   ```python
   import requests

   url = "http://<host>:<port>/<endpoint>"

   payload = {
       "input_field_1": "value",
       "input_field_2": "value"
   }

   headers = {
       "x-api-key": "your_api_key"
   }

   response = requests.post(url, json=payload, headers=headers)
   print(response.json())
   ```

2. **Payload File**
   A JSON file (`payload.json`) containing all input parameters with sample values.

3. **Output File**
   A JSON file (`output.json`) showing the expected response format with sample values.

4. **Ready-to-Run Script**
   A script (`run.py`) that reads `payload.json` as input, calls the API, and prints the output. Any team member should be able to execute it with a single command: `python run.py`

5. **Environment Variables Naming Convention**
   All API keys and secrets must be stored in an `.env` file. Variable names must be short, descriptive, and specific to that API so that when multiple APIs share a single `.env` file, each key is immediately identifiable. Avoid generic names like `API_KEY` or `KEY`. Use the format `FEATURE_PURPOSE`, for example: `TEXT_EXTRACTOR_KEY`, `FACE_DETECT_KEY`, `SENTIMENT_API_KEY`.

6. **API Key Strength**
   API keys used during development must be strong and unique. Do not use placeholder keys like `test_key`, `1234`, or `mykey`. Generate a proper key—a long random alphanumeric string or a passphrase-style string, for example: `scs@TextExtract#2026$secure` or a UUID. Weak keys must be replaced before submission.

7. **Example Environment File**
   An `example.env` file must be included in the submission. It should list all required environment variables with placeholder values so any developer can set up their environment quickly. The actual `.env` file with real keys must never be submitted.

8. **Clean File Structure**
   The API folder must be clean and organised. No unnecessary files, no temporary files, no test clutter, no unused scripts. Every file in the folder must serve a clear purpose.

9. **File Naming Convention**
   All files must have proper, meaningful names. Do not use names like `finalapi.py`, `final_one.py`, or `copy.py`. The main API file must be named `app.py` or `main.py` and all other files should clearly reflect their purpose.
