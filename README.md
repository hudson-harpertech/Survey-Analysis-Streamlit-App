# Text Anonymizer and Survey Thematic Coder 📊

**Author:** Hudson Harper  
**Date Released:** 3/27/2024

## Overview
This Streamlit app is designed for the anonymization of personally identifiable information (PII) within text and the subsequent coding of the text based on a predefined list of thematic codes and ratings. It enables users to process individual text entries or bulk-process via CSV file upload, streamlining the tasks of anonymizing and coding text data efficiently.

## Key Features
- **Anonymization of PII:** Utilizes spaCy for Named Entity Recognition to identify and anonymize names, organizations, locations, dates, times, email addresses, and phone numbers.
- **Text Coding with AI:** Leverages the OpenAI API and the gpt-3.5-turbo model for coding text according to custom thematic codes and ratings provided by the user.
- **Bulk Processing:** Supports uploading a CSV file for the anonymization and coding of multiple text entries at once.
- **Data Privacy:** Ensures that data is not retained nor used for any purposes beyond the intended anonymization and coding.

## Getting Started
1. **Access the App:** Visit [https://survey-ai-thematic-coder.streamlit.app/](https://survey-ai-thematic-coder.streamlit.app/) to start using the app.
2. **Enter your OpenAI API Key:** Necessary for the coding functionality. If you do not have an API key, create one by signing up at [platform.openai.com](https://platform.openai.com), navigating to "API keys" in the side menu, and clicking "+ Create new secret key".
3. **Add Codes and Ratings:** Customize the app by adding your own thematic codes and ratings to categorize the text.

## User Responsibility
Users are responsible for the results produced by this tool, including its responsible and ethical use. Review the anonymized text and applied codes to ensure accuracy and appropriateness for the analyzed text.

*Note: This app is designed with data privacy and ethical use in mind. It does not retain any user data beyond the session's scope.*
