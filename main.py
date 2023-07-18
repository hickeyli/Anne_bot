import streamlit as st
import os
import openai
import json
import re
import pandas as pd
import toml
import pinecone

st.set_page_config(layout="wide")


st.title('Annie: The ANR Assistant')

key = st.secrets['path']

if not key:
    st.write("Please input an OpenAPI key to continue")
else:
    openai.api_key = key  
    huggingface_key = "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"
    os.environ['OPENAI_API_KEY'] = key
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_key
    # Access the API key


    pinecone.init(      
        api_key='482713ab-eec5-432a-90d4-a16bcb867334',      
        environment='asia-northeast1-gcp'      
    )      
    index = pinecone.Index('kbchat')

    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores import Pinecone
    from langchain.document_loaders import UnstructuredFileLoader
    from langchain.chains import RetrievalQAWithSourcesChain
    from langchain import PromptTemplate, FewShotPromptTemplate
    from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
    from langchain import OpenAI

    llm = OpenAI(temperature=0)


    EMBEDDING_MODEL = "text-embedding-ada-002"
    GPT_MODEL = "gpt-3.5-turbo"

    ## Loader

    embeddings = HuggingFaceEmbeddings()
    index_name = 'kbchat'

    #db = FAISS.load_local("faiss_index", embeddings)
    db = Pinecone.from_existing_index('kbchat', embeddings)

    # Page layout
    info = ''
    summary = ''
    transcript = ''
    transcribed_files = set()


 


    # Sidebar
    with st.sidebar:
        st.header('Navigation')
        page = st.radio('Select a page', ['Voicemail Bot'])

        # Categorize Issues


    # Load data from Excel file
    df = pd.read_excel('issues.xlsx')

    def categorize_issue(issue):
        keywords = df['Keyword'].tolist()

        # # An example prompt with multiple input variables
        multiple_input_prompt = PromptTemplate(
            input_variables=['issue_test', 'keyword_test'], 
            template="Take this issue: {issue_test} and tell me which one of these keywords it most closely resembles: {keyword_test}. Respond with this format: \nKeyword:"
        )

        guess = multiple_input_prompt.format(issue_test = issue, keyword_test= keywords)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful IT assistant who is trying to detect the keyword the given problem most likely relates too. You are given the following text:"
                },
                {
                    "role": "user",
                    "content": guess
                }
            ],
            temperature=0.0,
            max_tokens=30  # Limit to approximately 20 words
        )

        keyword_response = response['choices'][0]['message']['content']

        # Extract the keyword from the response

        keyword = keyword_response.split(": ")[1]
        # Find the row in the DataFrame where the keyword column matches the given keyword
        matching_row = df[df['Keyword'] == keyword]
        if matching_row.empty:
            category = 'Desktop and Mobile Computing / Desktop and Mobile Device Support'
        else:
            category = matching_row['Category'].values[0]

        return category

    def extract_info_from_text(text):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Try to extract a netID, full name, phone number, ticket number and issue the user is experiencing from the following text. Keep in mind, the information you're looking for may not always be provided. Respond in the following format: \n\nNetID: \nName: \nPhone number: \nTicket number: \nIssue:"
                },
                {
                    "role": "user",
                    "content": text
                }
                
            ],
            temperature=0.0
        )
        data = """
            NetID: {netID}\n
            Name: {name}\n
            Phone Number: {phone_number}\n
            Ticket Number: {ticket_number}\n
            Issue: {issue}
        """
        # Generate context around the text using GPT-3
        generated_text = response['choices'][0]['message']['content']

        # Extract netID, phone number, and ticket number using regular expressions
        netID = re.search(r'NetID: (\w+)', generated_text)
        netID = netID.group(1) if netID else None

        phone_number = re.search(r'Phone number: (\d{3}-\d{3}-\d{4})', generated_text)
        phone_number = phone_number.group(1) if phone_number else None

        ticket_number = re.search(r'Ticket number: (\d+)', generated_text)
        ticket_number = ticket_number.group(1) if ticket_number else None

        # Extract name and issue using other methods (e.g., Named Entity Recognition)
        name = extract_name(generated_text)
        issue = extract_issue(generated_text)

        # Create a DataFrame from the extracted information

        formatted_data = data.format(netID=netID, name=name, phone_number=phone_number, ticket_number=ticket_number, issue=issue)
        st.write(formatted_data)

        data_table = {
            'NetID': [netID],
            'Name': [name],
            'Phone Number': [phone_number],
            'Ticket Number': [ticket_number],
            'Issue': [issue]
        }
        df = pd.DataFrame(data_table)
        

        return df

        # return netID, name, phone_number, ticket_number, issue

    def extract_name(text):
        # Split the text into lines
        lines = text.split('\n')

        # Look for a line that starts with "Name: "
        for line in lines:
            if line.startswith('Name: '):
                # Remove "Name: " and return the rest of the line
                return line[len('Name: '):]

        # If no line starts with "Name: ", return None
        return None


    def extract_issue(text):
        # Split the text into lines
        lines = text.split('\n')

        # Look for a line that starts with "Issue: "
        for line in lines:
            if line.startswith('Issue: '):
                # Remove "Issue: " and return the rest of the line
                return line[len('Issue: '):]

        # If no line starts with "Issue: ", return None
        return None




    # # VM Bot KB Check


    # def knowledgebase_search(type, issue):
    #     from langchain.chains import RetrievalQA
    #     if type == 'vm_kb':

    #         problem = summarize_text(issue)
    #         st.write(problem)
    #         prompt_template = """Is there any relevant information in the knowledge base that can help with this issue? If so, please provide the solution. If not, please respond with "No relevant information in knowledge base." 

    #         Issue: %s 

    #         Solution:""" % problem
    #     else:
    #         prompt_template = issue
        
    #     qa = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff", retriever=db.as_retriever())
        

    #     result = qa({"question": prompt_template}, return_only_outputs=True)
    #     #qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=db.as_retriever())

    #     #result = qa.run(prompt_template)
        
    #     return result

    def summarize_text(issue):
        issue_dict = issue.to_dict()
        text_string=json.dumps(issue_dict)


        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful IT assistant who is trying to detect the most likely issue the user is experiencing. You are given the following text:"
                },
                {
                    "role": "user",
                    "content": text_string
                },
                {
                    "role": "assistant",
                    "content": "Please summarize the above text in less than 10 words."
                }
            ],
            max_tokens=30  # Limit to approximately 20 words
        )

        summary = response['choices'][0]['message']['content']
        return summary

    def transcribe_audio(file):
        transcription = openai.Audio.transcribe("whisper-1", file, prompt = 'You are transcribing a phone call about a computer problem someone is experiencing', response_format = 'text')
        return transcription

    def ticket_definer(info):
        if info is not None:
            #st.write(info)
            st.header('Subject')
            if info['Issue'] is not None:
                ticket_summary = summarize_text(info['Issue'])
                st.write(ticket_summary)
                #st.header('Description')
                #st.write(str(info["Issue"]))
                #st.write(info['Issue'])
                st.header('Category')
                category = categorize_issue(ticket_summary)
                st.write(category)
                definition = determine_priority(info['Issue'])
                #st.write(ticket_info)
                st.header('Form')
                st.write(definition)
        else:
            st.write('No information extracted.')

    def determine_priority(issue):
        high_priority_keywords = df['Keyword'].tolist()

        # # An example prompt with multiple input variables
        multiple_input_prompt = PromptTemplate(
            input_variables=['issue_test', 'incident_list'], 
            template="Take this issue: {issue_test} and decide if it closesy resembles on of the following descriptions of an issue: {incident_list}. If the issue given relates to one of the definitions, then respond with this \nDefinition: Incident. If it doesn't match, then respond with \n Definition: Service Request."
        )

        guess = multiple_input_prompt.format(issue_test = issue, incident_list= high_priority_keywords)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful IT assistant who is trying to decide if the issue given is considered a high priority issue. You are given the following text:"
                },
                {
                    "role": "user",
                    "content": guess
                }
            ],
            temperature=0.0,
            max_tokens=30  # Limit to approximately 20 words
        )

        priority_response = response['choices'][0]['message']['content']

        # Extract the keyword from the response

        priority = priority_response.split(": ")[1]
        # Find the row in the DataFrame where the keyword column matches the given keyword
        
        if priority == 'Incident':
            form = 'Incident Form'
        else:
            form = 'Service Request Form'

        return form


    # def delete_files(folder_path):
    #     # Get the list of files in the folder
    #     file_list = os.listdir(folder_path)

    #     # Iterate over the files and delete them
    #     for file_name in file_list:
    #         file_path = os.path.join(folder_path, file_name)
    #         os.remove(file_path)


    # Page content
    if page == 'Voicemail Bot':
        st.header('Voicemail Bot')
        st.markdown('Let me help you out')

        #col1, col2 = st.columns(2)

        #with col1:
        uploaded_file = st.file_uploader(label='Upload voicemail file.', type ='.wav', accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        if uploaded_file is not None:
            if transcript is not None:
                ## Transcribes audio file
                transcript = transcribe_audio(uploaded_file)
                
                st.header('Transcript')
                st.write(transcript)
                st.header('Information')
                info = extract_info_from_text(transcript)
                ticket_info = ticket_definer(info)
                
                st.write(ticket_info)
                #st.write(info)
        else:
            st.write('No file uploaded.')              

                    
        # with col2:
        #     helper = st.checkbox('Want me to search the knowledge base?')
        #     if helper:
        #         type = 'vm_kb'
        #         prediction = knowledgebase_search(type, info['Issue'])
        #         # Load JSON data
        #         solution = prediction

        #         # Get the 'answer' and 'sources' values
        #         answer = solution.get('answer', 'No answer provided.')
        #         sources = solution.get('sources', 'No sources provided.')

        #         # Print the information
        #         output = (f"Answer:\n{answer}\n\nSources:\n{sources}")
        #         st.header('Possible Solution')
        #         st.write(output)

                


    # elif page == 'Folder Watcher':
        

    #     import time
    #     import io
    #     from pathlib import Path

    #     st.title("Folder Cleanup")
        
    #     folder_path = "C:/Users/hickeyli/OneDrive - Michigan State University/Desktop/AI/VMTranscribe/VMs"  # Replace with the actual folder path

    #     if st.button("Delete Folder Contents"):
    #         delete_files(folder_path)
    #         st.write("Folder contents deleted.")
        
    #     st.header('Folder Watcher')

    #     def monitor_directory(path, check_interval):
    #         directory_path = Path(path)
    #         known_files = None # initial set of files in directory
    #         #transcribed_files = set() 

    #         while True:
    #             known_files, transcribed_files = check_for_new_files(directory_path, known_files)
    #             time.sleep(check_interval)  # wait for the specified interval before checking again`


    #     def check_for_new_files(directory_path, known_files):
    #         current_files = set(directory_path.glob('*'))  # get current set of files
    #         if known_files is None:
    #             known_files = set()

    #         new_files = current_files - known_files  # find the difference between the current and known files
            
    #         for new_file in new_files:
    #             file_path = Path(new_file)
    #             if str(file_path) not in transcribed_files:
    #                 with open(file_path, 'rb') as audio_file:
    #                     audio_bytes = audio_file.read()
    #                     audio_file_like = io.BytesIO(audio_bytes)
    #                     audio_file_like.name = str(file_path) 
    #                     transcript_newfile = transcribe_audio(audio_file_like)
    #                     transcribed_files.add(file_path)
                
    #             st.header('Transcript')
    #             st.write(transcript_newfile)
    #             st.header('Information')
    #             info = extract_info_from_text(transcript_newfile)
    #             ticket_info = ticket_definer(info)
    #             definition = determine_priority(info)
    #             st.write(ticket_info)
    #             st.header('Form')
    #             st.write(definition)  


    #         return current_files, transcribed_files
        
    #             # You can add your transcription code here.
    #             # return the current set of files as the new known_files for the next iteration


        
    #     monitor_directory('C:/Users/hickeyli/OneDrive - Michigan State University/Desktop/AI/VMTranscribe/VMs', 10)  # checks the directory every 10 seconds


        

    # elif page == 'KB Chat':
    #     st.header('KB Chat')
    #     label = 'What do you want to know?'
    #     query = st.text_input(label, value="", max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")
    #     if query:
    #         type = 'kb_search'
    #         solution = knowledgebase_search(type, query)
    #         st.header('This is what I found')

    #         chat_answer = solution

    #             # Get the 'answer' and 'sources' values
    #         answer = chat_answer.get('answer', 'No answer provided.')
    #         sources = chat_answer.get('sources', 'No sources provided.')

    #         # Print the information
    #         chat_output = (f"Answer:\n{answer}\n\nSources:\n{sources}")
    #         st.write(chat_output)





    ## folder_to_monitor = 'C:/Users/hickeyli/OneDrive - Michigan State University/Desktop/AI/VMTranscribe/VMs'



st.markdown("""<style>.reportview-container .main .block-container{max-width: 800px;}</style>""", unsafe_allow_html=True)
## Loader












