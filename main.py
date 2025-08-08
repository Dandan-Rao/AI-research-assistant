#!/usr/bin/env python
import os
import openai
import param

import panel as pn  # GUI

# Load Panel's interactive features
pn.extension()

# Set openai api key
openai.api_key  = os.environ['OPENAI_API_KEY']
if openai.api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS # A more persistent vector store option
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
import shutil
CHAIN_TYPE = "stuff"  # Default chain type for ConversationalRetrievalChain

# Ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._migration")

# Set the openai chat model name
llm_name = 'gpt-4o-mini'
# define embedding
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
# Define text splitter. Splitting text into smaller chunks while maintaining the context. RecursiveCharacterTextSplitter is recommended for generic text.
# 1000-1500 is a moderate chunk_size. The chunk size is 1024, which is a good balance between speed and accuracy.
# Seperate by paragraphs, lines, setences, spaces, and empty strings.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=96) # The default list is ["\n\n", "\n", " ", ""]

# Initialize the vector store
DB_PATH = os.path.join(".", "docs", "chroma") 
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)
vectordb = Chroma(
    collection_name = 'Self-Assistant',
    persist_directory=DB_PATH,  # Where to save data locally
    embedding_function=embeddings
    )
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})  # Default to 4 results

def load_db(file_path):
    '''
    Load the file, split the text, add document to a vector store, save the vector store, and create a chatbot chain.
    '''
    # global db  # Ensure db persists
    # Load the document
    if not file_path:
        return
    elif file_path.endswith(".pdf"):
        document = PyPDFLoader(file_path).load()
    elif file_path.endswith(".txt"):
        document = TextLoader(file_path).load()
    elif file_path.endswith(".docx"):
        document = UnstructuredWordDocumentLoader(file_path).load()
    elif file_path.startswith("http"):
        document = WebBaseLoader(file_path).load()
    else:
        print(f"Error loading file {file_path}: Unsupported file type.")
        return

    # split documents
    docs = text_splitter.split_documents(document)

    # Add documents to the vector store if they exist
    if docs:
        vectordb.add_documents(docs)  # Add new documents
        # vectordb.persist()  # ensures that new data is stored in DB_PATH


# create classes with dynamic parameters
class cbfs(param.Parameterized): 
    '''
    Chatbot for files and web links
    '''
    chat_history = param.List([])
    answer = param.String("")
    db_query  = param.String("")
    db_response = param.List([])
    def __init__(self, k = 4, **params):
        super(cbfs, self).__init__(**params)
        self.panels = []
        self.loaded_files = [] # record loaded files
    
    # load files and create chatbot chain, manage memory
    def call_load_db(self, count):
        if count == 0 or not file_input.value and not url_input.value.strip():  # init or no file specified :
            return pn.pane.Markdown("No file or URL uploaded.")
        warning_message = ""
        if file_input.value:  # If user uploads a file
            self.loaded_files.extend(file_input.value)
            warning_message += "⚠️ **The following files have been added:**\n- " + "\n- ".join(file_input.value) + "\n\n"
            # upload_button.button_style="outline"
        if url_input.value.strip():  # If user enters a web link
            self.loaded_files.append(url_input.value.strip())
            warning_message += "🌍 **The following URL has been added:**\n- " + url_input.value.strip() + "\n\n"
        try:
            for file_path in self.loaded_files:
                self.chatbot_chain = load_db(file_path)
                upload_button.button_style="solid"
            self.loaded_files = []  # Clear the list of loaded files
            # return pn.pane.Alert(f"Loaded File: {self.loaded_files}")
            return pn.pane.Alert(warning_message)
        
        except Exception as e:
            return pn.pane.Alert(f"Error loading files: {e}")

    def convchain(self, query):
        if not query:
            return pn.WidgetBox(
                pn.pane.Markdown(
                    "#### 💡 Ask a Question!  \n"
                    "You can ask questions about the files you uploaded in the **'Configure'** tab. 📂",
                    width=600
                    )
                )
        if vectordb._collection.count() == 0:
            return pn.pane.Markdown("The database is empty. Please upload a file or enter a URL in the 'Configure' tab.")
        
        chatbot_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name=llm_name, temperature=0), 
            chain_type=CHAIN_TYPE, 
            retriever=retriever, 
            return_source_documents=True,
            return_generated_question=True,
        )
        # invoke chatbot chain
        result = chatbot_chain.invoke({"question": query, "chat_history": self.chat_history})

        # updata chat histroy and other state
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer'] 
        
        # Add the new conversation to the panels
        new_exchange = [
            pn.Row('User:', pn.pane.Markdown(query, width=600)),
            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=600, styles={'background-color': '#F6F6F6'}))
            ]
        
        self.panels.extend(new_exchange)

        # question_input.value = ''  # clears loading indicator when cleared
        return pn.WidgetBox(*self.panels, scroll=True)

    @param.depends('db_query', )
    def get_lquest(self):
        if not self.db_query :
            return pn.Column(
                pn.Row(pn.pane.Markdown("### No Database Queries Yet", styles={'background-color': '#F6F6F6'})),
                pn.layout.Divider(),
                pn.Row(pn.pane.Str("You haven't asked any questions that required database retrieval."))
                )
        return pn.Column(
            pn.Row(pn.pane.Markdown("### Last Database Query", styles={'background-color': '#F6F6F6'})),
            pn.layout.Divider(),
            pn.pane.Str(f"🔍 {self.db_query}")
            )

    @param.depends('db_response', )
    def get_sources(self):
        if not self.db_response:
            return 
        rlist=[pn.Row(pn.pane.Markdown(f"Result of DB lookup:", styles={'background-color': '#F6F6F6'}))]
        for doc in self.db_response:
            rlist.append(pn.Row(pn.pane.Str(doc)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    @param.depends('convchain', 'clr_history') 
    def get_chats(self):
        if not self.chat_history:
            return pn.WidgetBox(pn.Row(pn.pane.Str("No Chat History Yet")), width=600, scroll=True)
        
        chat_display=[pn.Row(pn.pane.Markdown(f"## Chat History", styles={'background-color': '#F6F6F6'}))]
    
        # Add each exchange in a more readable format
        for i, (question, answer) in enumerate(self.chat_history, 1):
            chat_display.extend([
                pn.Row(pn.pane.Markdown(f"**Question {i}:** {question}")),
                pn.Row(pn.pane.Markdown(f"**Answer {i}:** {answer}")),
                pn.layout.Divider()  # Add visual separation between exchanges
            ])
        return pn.WidgetBox(*chat_display, width=600, scroll=True)

    def clr_history(self, count = 0):
        self.chat_history = []
        self.panels = []
        return 
    
    def clr_db(self, count = 0):
        # Remove all files in the vector store directory
        print(f'{count}-------------')
        if count == 1 and os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
            print('--------clear---------')
            os.makedirs(DB_PATH, exist_ok=True)  # Recreate an empty directory
        self.chatbot_chain = load_db(None)

cb = cbfs()

# Define UI elements
## function: file Selector (Accepts PDF, TXT, DOCX)
file_input = pn.widgets.FileSelector(directory='.')
# Text input for web links
url_input = pn.widgets.TextInput(placeholder="Enter web link here…", sizing_mode='stretch_width')

upload_button = pn.widgets.Button(name="Upload & Process", button_type="primary")
bound_button_load = pn.bind(cb.call_load_db, upload_button.param.clicks)

## Function to display selected files
def display_selected_files(event):
    return pn.pane.Markdown(f"### Uploaded Files:\n- " + "\n- ".join(file_input.value))

## function: clear history
button_clearhistory = pn.widgets.Button(name="Clear Chat History", button_type='warning')
button_clearhistory.on_click(cb.clr_history)

## function: clear database
button_cleardb = pn.widgets.Button(name="Clear Database", button_type='warning')
button_cleardb.on_click(cb.clr_db)

## function: input question
question_input = pn.widgets.TextInput(
    placeholder='💬 Ask your question here...',
    width=600,  # Increase width to make it more prominent
    sizing_mode='stretch_width',  # Make it expand with the layout
    css_classes=['input-highlight'],  # Custom class for styling
)


# UI Layout
# Define Layout
tab1 = pn.Column(
    pn.Row(question_input),
    pn.layout.Divider(), ## used for spacing, visual seperation
    # pn.panel(conversation, loading_indicator=True, height=300),
    pn.bind(cb.convchain, question_input),
    pn.layout.Divider(),
)

tab2= pn.Column(
    pn.Row(button_cleardb, pn.pane.Markdown("Clear all memory and start fresh.")),
    pn.panel(cb.get_lquest),
    pn.layout.Divider(),
    pn.panel(cb.get_sources),
)
tab3= pn.Column(
    pn.Row(button_clearhistory, pn.pane.Markdown("Clears chat history. Can use to start a new topic")),
    pn.panel(cb.get_chats),
    pn.layout.Divider(),
)

tab4 = pn.Column(
    pn.pane.Markdown("## Upload Multiple Files or URL"),
    file_input,
    url_input,
    pn.bind(display_selected_files, file_input),  # Dynamically updates file display
    upload_button,  # The new button that uploads and triggers processing
    bound_button_load,
    pn.layout.Divider(),
)

dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('# Chat With Your Files: AI Research Assistant', styles={'text-align': 'center', 'font-size': '1em', 'color': "#3B7FE6"})),
    pn.Tabs(('Conversation', tab1), ('Database', tab2), ('Chat History', tab3),('Configure', tab4))
)

debug_mode = True  # Set debug mode to True for development

if __name__ == "__main__":
    # Option 1: Use pn.serve directly
    pn.serve(dashboard, show=True, title="My Dashboard", autoreload=debug_mode)

