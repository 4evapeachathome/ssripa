import azure.functions as func
import json
from rag_answer import generate_consolidated_answer

app = func.FunctionApp()

@app.route(route="rag_query")
@app.function_name("rag_query")
def rag_query(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # Get request body
        req_body = req.get_json()
        
        # Validate request body
        if not req_body or 'questions' not in req_body:
            return func.HttpResponse(
                json.dumps({"error": "Please provide 'questions' array in the request body"}),
                mimetype="application/json",
                status_code=400
            )
        
        questions = req_body['questions']
        if not isinstance(questions, list):
            return func.HttpResponse(
                json.dumps({"error": "'questions' must be an array"}),
                mimetype="application/json",
                status_code=400
            )

        # Generate consolidated answer
        answer = generate_consolidated_answer(questions)
        
        # Return response
        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "answer": answer
            }),
            mimetype="application/json"
        )
        
    except ValueError as ve:
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON in request body"}),
            mimetype="application/json",
            status_code=400
        )
    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
