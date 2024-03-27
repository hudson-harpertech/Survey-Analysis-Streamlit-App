import plotly.express as px
import numpy as np
import streamlit as st
from annotated_text import annotated_text
import spacy
import re
import json
import pandas as pd
from openai import OpenAI
import os

# Load the NLP model
nlp = spacy.load("./spacy_model/en_core_web_sm/en_core_web_sm-3.0.0")

# Set up the session state
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""
    st.session_state["client"] = None

if "anonymized_text" not in st.session_state:
    st.session_state["anonymized_text"] = ""

if "codes_output" not in st.session_state:
    st.session_state["codes_output"] = {}

if "codes" not in st.session_state:
    st.session_state["codes"] = []

if "ratings" not in st.session_state:
    st.session_state["ratings"] = []

if "codes_df" not in st.session_state:
    st.session_state["codes_df"] = None

if "df" not in st.session_state:
    st.session_state["df"] = None

if "properties" not in st.session_state:
    st.session_state["properties"] = {}

if "selected_column" not in st.session_state:
    st.session_state["selected_column"] = None


# Set up the properties for the OpenAI function
st.session_state["properties"] = {code: {"description": f"Assign the appropriate rating for the following code: {code}. Be sure to use the N/A option if the code does not apply to the text.",
                                         "type": "string", "enum": st.session_state["ratings"] + ['N/A']} for code in st.session_state["codes"]}

# Set up the functions for the OpenAI API
my_functions = [
    {
        "name": "encode_text",
        "description": "Function that encodes text based on a list of codes.",
        "parameters": {
            "type": "object",
            "properties": st.session_state['properties'],
            "required": st.session_state["codes"]
        }
    }
]

# Functions
    
def anonymize_text(text: str) -> str:
    """
    Anonymizes sensitive information in the given text.

    Args:
        text (str): The text to be anonymized.

    Returns:
        str: The anonymized text with sensitive information replaced by placeholders.

    """
    # Using spaCy for Named Entity Recognition
    doc = nlp(text)

    # Storing the indices of entities to anonymize
    pii_entities = []
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "EMAIL", "PHONE_NUMBER"]:
            pii_entities.append((ent.start_char, ent.end_char))

    # Anonymize the entities
    anonymized_text = ""
    prev_end = 0
    for start, end in pii_entities:
        anonymized_text += text[prev_end:start] + "|PII|"
        prev_end = end
    anonymized_text += text[prev_end:]

    # Regular expressions for anonymizing email and phone numbers
    email_pattern = r'\S+@\S+'
    phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'

    # Replacing email and phone numbers
    anonymized_text = re.sub(email_pattern, '|EMAIL|', anonymized_text)
    anonymized_text = re.sub(phone_pattern, '|PHONE|', anonymized_text)

    return anonymized_text


def create_annotate_PII_input(text: str) -> list:
    """
    Creates an annotated PII (Personally Identifiable Information) input by replacing specific words with empty strings.

    Args:
        text (str): The input text to be processed.

    Returns:
        list: A list of words where specific PII-related words are replaced with empty strings.

    Example:
        >>> create_annotate_PII_input("This is a sample text|containing|PII|information")
        ['This is a sample text', 'containing', ('PII', ''), 'information']
    """
    split_text = text.split("|")
    for i, word in enumerate(split_text):
        if word in ["PII", "PHONE", "EMAIL"]:
            split_text[i] = (word, "")
    return split_text


def set_anonymized_text() -> None:
    """
    Sets the anonymized text in the session state by calling the `anonymize_text` function.

    Parameters:
        None

    Returns:
        None
    """
    st.session_state["anonymized_text"] = anonymize_text(
        st.session_state["text_input"])

def get_codes() -> None:
    """
    Retrieves codes for anonymized text using a chatbot model and sets them to the session state.

    Returns:
        str: The generated codes for the anonymized text.
    """
    if "anonymized_text" == "":
        return ""

    response = st.session_state['client'].chat.completions.create(model="gpt-4-0125-preview",
                                              messages=[
                                                  {"role": "system", "content": "You are an institutional researcher who is coding survey responses from parents about their child's high school experience."},
                                                  {"role": "user",
                                                   "content": f"Code the following text: {st.session_state['anonymized_text']}"}
                                              ],
                                              functions=my_functions)

    st.session_state['codes_output'] = json.loads(
        response.choices[0].message.function_call.arguments)

def get_df_codes(text: str) -> dict:
    """
    Retrieves the codes for the given text by using the OpenAI GPT-4 model.

    Parameters:
        text (str): The text to be coded.

    Returns:
        dict: A dictionary containing the codes for the given text.

    Raises:
        KeyError: If the 'client' key is not found in the session state.

    """
    response = st.session_state['client'].chat.completions.create(model="gpt-4-0125-preview",
                                              messages=[
                                                  {"role": "system", "content": "You are an institutional researcher who is coding survey responses from parents about their child's high school experience."},
                                                  {"role": "user",
                                                   "content": f"Code the following text: {text}"}
                                              ],
                                              functions=my_functions)

    return json.loads(response.choices[0].message.function_call.arguments)

def append_codes() -> None:
    """
    Appends the code input to the list of codes in the session state.

    If the code input is not already in the list of codes, it is appended and the list is sorted.

    Parameters:
        None

    Returns:
        None
    """
    if st.session_state["code_input"] not in st.session_state["codes"]:
        st.session_state["codes"].append(st.session_state["code_input"])
        st.session_state["codes"] = sorted(st.session_state["codes"])
    st.session_state["code_input"] = ""


def append_ratings() -> None:
    """
    Appends the rating input to the ratings list in the session state.

    If the rating input is not already in the ratings list, it is added and the ratings list is sorted.

    Parameters:
        None

    Returns:
        None
    """
    if st.session_state["rating_input"] not in st.session_state["ratings"]:
        st.session_state["ratings"].append(st.session_state["rating_input"])
        st.session_state["ratings"] = sorted(st.session_state["ratings"])
    st.session_state["rating_input"] = ""

def submit_code():
    """
    This function is responsible for submitting the code.
    It calls the `append_codes` function to append the codes.
    """
    append_codes()

def submit_rating():
    """
    This function is responsible for submitting a rating.
    It calls the `append_ratings` function to append the rating to the ratings list.
    """
    append_ratings()

# Introduction Section
st.set_page_config(layout="wide")
st.title(':flags: Text Anonymizer and Survey Thematic Coder :bar_chart:')
st.markdown("##### Author: [Hudson Harper](https://www.linkedin.com/in/hudsonharper/)\n##### Date Released: 3/27/2024")

st.markdown("This app allows you to anonymize personally identifiable information (PII) in text and then code the text based on a list of codes and ratings. You can also upload a CSV file to anonymize and code multiple text entries at once.")

st.markdown("This app uses the spaCy library for Named Entity Recognition and the OpenAI API with the gpt-3.5-turbo model to code the text. The app is designed to anonymize PII such as names, organizations, locations, dates, times, email addresses, and phone numbers. The app also allows you to add custom codes and ratings to categorize the text. Data is not retained by the app nor is it used for any purpose other than the intended use.")

st.markdown("The user is responsible for the results of using this tool including its responsible and ethical use. The user should review the anonymized text and codes to ensure that the text is anonymized correctly and the codes are applied accurately. The user should also review the codes and ratings to ensure that they are appropriate for the text being analyzed.")

st.markdown("#### Enter your OpenAI API Key and add codes and ratings to get started.")
st.markdown("To create an API key, go to [platform.openai.com](https:'//platform.openai.com/) and create an account. Then, navigate to API keys in the side menu. Finally, click + Create new secret key and copy the generated API key.")

# OpenAI API Key Input
openai_api_key = st.text_input('OpenAI API Key', '<YOUR_API_KEY>')
submit_openai_api_key_button = st.button("Submit", on_click=lambda: st.session_state.update(
    {"openai_api_key": openai_api_key, "client": OpenAI(api_key=openai_api_key)}))

# Main App Section
if st.session_state["openai_api_key"] != "":
    with st.container():
        left, middle, right = st.columns([6, 1, 6])
        with left:
            st.markdown(
                "#### Add Codes Here")
            st.text_input('Add codes one at a time and then press "Add"',
                        key="code_input",
                        on_change=submit_code)
            add_code_button = st.button(
                "Add", on_click=append_codes, key="add_code_button")
            st.write("__Current Codes:__ ", ', '.join(st.session_state["codes"]))

        with middle:
            st.empty()

        with right:
            st.markdown(
                "#### Add Ratings Here")
            st.text_input('Add ratings one at a time and then press "Add"',
                        key="rating_input",
                        on_change=submit_rating)
            add_rating_button = st.button(
                "Add", on_click=append_ratings, key="add_rating_button")
            st.write("__Current Ratings:__ ", ', '.join(
                st.session_state["ratings"]))

    st.divider()

    with st.container():
        col1, col2, col3, col4, col5 = st.columns([6, 1, 6, 1, 6])

        with col1:
            st.markdown("### Original Text:")
            text = st.text_area(
                "### Enter text to anonymize and code here", key="text_input", on_change=set_anonymized_text)
            anonymize_button = st.button("Anonymize Text",
                                        on_click=set_anonymized_text)

        with col2:
            st.empty()

        with col3:
            st.markdown("### Anonymized Text:")
            # Anonymize the text
            if st.session_state['anonymized_text'] != "":
                anonymized_text = anonymize_text(text)
                st.write("")
                annotated_text(create_annotate_PII_input(anonymized_text))
                code_button = st.button("Code Text", on_click=get_codes)

        with col4:
            st.empty()

        with col5:
            st.markdown("### Codes and Ratings:")
            if st.session_state['codes_output'] != {}:
                st.write(st.session_state['codes_output'])

    st.divider()


    def process_csv_file(column_name: str = "Comments") -> None:
        if st.session_state['df'] is None:
            return  
        df = st.session_state['df'].copy().dropna()
        df['anonymized_text'] = df[column_name].astype(str).apply(anonymize_text)
        df['codes'] = df['anonymized_text'].astype(str).apply(get_df_codes)

        st.session_state['codes_df'] = pd.DataFrame(df['codes'].tolist())


    with st.container():
        st.markdown("### CSV Anonymizer and Coder")

        col1, col2, col3, col4, col5 = st.columns([6, 1, 6, 1, 6])

        with col1:
            st.markdown("#### Upload CSV File")
            csv_file = st.file_uploader(
                "Upload CSV file here", type=["csv"], key="csv_file")
            if st.session_state['csv_file'] is not None:
                process_button = st.button(
                    "Process CSV File", on_click=process_csv_file(st.session_state["selected_column"]))

        with col2:
            st.empty()

        with col3:
            st.markdown("#### CSV File Data")
            if st.session_state['csv_file'] is not None:
                st.session_state['df'] = pd.read_csv(st.session_state['csv_file'])
                st.dataframe(
                    st.session_state['df'].dropna(), use_container_width=True, hide_index=True)
                st.selectbox("Select Column to Analyze", 
                             st.session_state['df'].columns, 
                             key="selected_column",
                             on_change=lambda: st.session_state.update({"selected_column": st.session_state["selected_column"]}),
                             )

        with col4:
            st.empty()

        with col5:
            st.markdown("#### Codes")
            if st.session_state['codes_df'] is not None:
                st.dataframe(st.session_state['codes_df'],
                            use_container_width=True, hide_index=True)
        if st.session_state['codes_df'] is not None:
            chart_df = st.session_state['codes_df'].replace("N/A", np.NaN).melt().sort_values('variable').groupby(['variable', 'value']).size().reset_index(
                name='count')

            chart_df.columns = ['Code', 'Rating', 'Count']

            chart = px.bar(chart_df,
                        x='Code',
                        y='Count',
                        color='Rating',
                        barmode='group',
                        title='Code Counts by Rating')

            st.plotly_chart(chart, use_container_width=True)
else:
    st.write("Please enter your OpenAI API Key")
