import azure.functions as func
import logging
import json

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        return func.HttpResponse(
            json.dumps({"message": "Hello from Azure Functions!"}),
            mimetype="application/json"
        )
    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
