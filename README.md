# Building a Semantic Search and QA System with LangChain, ChromaDB, and Google Generative AI
This guide walks you through setting up a semantic search and question-answering system using LangChain, ChromaDB, and Google's Generative AI models. We'll load documents, create embeddings, build a vector database, and interact with the model to retrieve information.

# Table of Contents
  1) Installation
  2) Data Preparation
  3) Setting Up the Environment
  4) Loading Documents
  5) Text Splitting
  6) Creating the Vector Database
  7) Semantic Search
  8) Building the QA Chain
  9) Interacting with Google Gemini

# 1) Installation:
  !pip -q install langchain chromadb google-generativeai langchain_google_genai langchain-community
  # Check the installed version of LangChain:
    !pip show langchain
  
# 2) Data Preparation:

  # Download and unzip the dataset containing articles:
    !wget -q https://www.dropbox.com/s/vs6ocyvpzzncvwh/new_articles.zip
    !unzip -q new_articles.zip -d new_articles
  
# 3) Setting Up the Environment:

  # Retrieve your Google API Key securely using Google Colab's userdata:
    from google.colab import userdata
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    
# 4) Loading Documents:

  # Import the necessary modules and load documents from the specified directory:
  from langchain.vectorstores import Chroma
  from langchain_google_genai import GoogleGenerativeAIEmbeddings
  from langchain.document_loaders import DirectoryLoader, TextLoader
  
  loader = DirectoryLoader('./new_articles/', glob="./*.txt", loader_cls=TextLoader)
  documents = loader.load()
  # View the loaded documents:
    documents

# 5) Text Splitting:

  # Split the loaded documents into manageable chunks for processing:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
  # Inspect the split texts:
    texts
  # View a specific text chunk (e.g., the fourth chunk):
    texts[3]
  
# 6) Creating the Vector Database:

  # Initialize the Embeddings and Vector Store:
    persist_directory = 'db'
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )

  # Persist the Database to Disk:
  
    # Persist the vector database to disk
      vectordb.persist()
      vectordb = None
    # Reload the Persisted Vector Database:
      vectordb = Chroma(
          persist_directory=persist_directory,
          embedding_function=embeddings
      )

# 7) Semantic Search:

  # Set up the retriever to perform semantic searches on the vector database:
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
  # Check the retriever's search type and parameters:
    retriever.search_type
    retriever.search_kwargs
    
# 8) Building the QA Chain:

  # Import Necessary Modules:
    from langchain.chains import RetrievalQA
    from langchain_google_genai import ChatGoogleGenerativeAI
    
  # Initialize the Chat Model:
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    
  # Create the QA Chain:
    qa_chain = RetrievalQA.from_chain_type(
        model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
      )
  
  # Define a Function to Process LLM Responses:
  
    # Display the result and the sources from the LLM response
      def process_llm_response(llm_response):
          print(llm_response['result'])
          print('\n\nSources:')
          for source in llm_response["source_documents"]:
              print(source.metadata['source'])
          
  # Perform a Sample Query:
    query = "what is GenerativeAI?"
    llm_response = qa_chain(query)
    process_llm_response(llm_response)
  
##################################################################################  

# 9) Interacting with Google Gemini:

  Use the invoke method to directly interact with the Google Gemini model:
  # Pose a question to the model
  response = model.invoke("What is Deep Learning?")
  
  # Display the model's response
  print("Model Response:", response.content)
