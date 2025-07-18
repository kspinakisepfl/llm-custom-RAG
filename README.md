# RAGbot setup

## How to setup the necessary dependencies

1. Open the .env file in the backend and place in it your API key for OpenAI
2. Create a conda envirnoment running python 3.11
3. Activate the environment and run `pip install -r requirements.txt` to install the required dependencies 
2. Make sure `npm` is installed on your system and that you are able to run the `npx` command in your commandline


## How to initiate the app

1. Add the content of your choice (PDF file format) to the data folder. The folder is currently filled with documents outlining EPFL's legal rules.
2. Run the `create_database.py` file to create the RAG database from the PDF files. If a database exists already, delete that one to make a new one.
3. Start the backend of the app by executing `python server.py` from the backend directory
4. Start the frontend of the app by running the command `npx http-server` in the command line from the frontend directory
5. Navigate to http://localhost:8080/ to utilize the app
