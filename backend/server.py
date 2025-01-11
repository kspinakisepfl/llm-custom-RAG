from flask import Flask, request, jsonify
import flask
import json
from assistant import parametrization, replace_with_sources, get_db_loc, process_pdf, create_sqlite_tables
from langchain.memory import ConversationBufferMemory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


@app.route('/chat', methods=["GET","POST"])
def chat():
    if request.method == "POST":
        received_data = request.get_json()
        print(f"received data: {received_data}")    # prints in the server-side console
        userquestion = received_data['data']
        ag_exec = parametrization(model = received_data["modelpick"],temp = received_data['temp'],top_p = received_data["top_p"], max_toks = received_data["maxtokens"])
        agentreply = ag_exec.invoke({"input": userquestion, "chat_history": memory.load_memory_variables({})["chat_history"]})
        print(agentreply)
        memory.save_context({"input": agentreply['input']}, {"output":agentreply['output']})
        agentreply['output'] = replace_with_sources(agentreply['output'])
        return_data = {
            "status":"success",
            "message":f"{agentreply['output']}"
        }
        return flask.Response(response=json.dumps(return_data),status=201)  # guessing we use .dumps here because of the f"{}" format employed above


if __name__ == '__main__':
    app.run("localhost", 7000, debug=True)