# RAGbot setup

## How to setup the necessary dependencies

1. Open the .env file and place in it your API key for OpenAI
2. Run `pip install -r requirements.txt` in your environment of choice
2. Make sure `npm` is installed on your system and that you are able to run the `npx` command in your commandline

## How to initiate the model

1. Add the content of your choice (PDF file format) to the data folder. The folder is currently filled with documents outlining EPFL's legal rules.
2. Run the create_database.py file to create the RAG database from the PDF files
3. Start the backend of the app by running the server.py file
4. Start the frontend of the app by running the command `npx http-server` in the command line
5. Navigate to http://localhost:8080/ to utilize the app