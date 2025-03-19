from django.http import HttpResponse, JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
from pymongo import MongoClient
from django.conf import settings
from bson import ObjectId
import traceback
import datetime
from datetime import timedelta
from collections import defaultdict

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from pymongo import MongoClient
import json
import re
import requests
import stripe
import pandas as pd
import random
import string
import hashlib
from dateutil.relativedelta import relativedelta

client = MongoClient(f'{settings.MONGO_URI}')
db = client['Limeblock']
users_collection = db['Users']
frontend_collection = db['Frontend']
backend_collection = db['Backend']

@csrf_exempt
def main(req):
    return HttpResponse("Wsg")

@csrf_exempt
def create_user(request):   
    try:
        # Parse the incoming JSON request body
        data = json.loads(request.body)
        emails = data.get("emails", [])
        print(emails)
        business_name = data.get("business_name")
        password = data.get("password")

        # Validate the request data
        if not business_name or not password or not emails or len(emails) == 0:
            return JsonResponse(
                {"message": "Invalid payload: Missing business_name or password or email address"},
                status=400
            )

        # Check if the user already exists
        existing_user = users_collection.find_one({"business_name": business_name})
        if existing_user:
            print("Name taken")
            return JsonResponse(
                {"warning": "Business name already taken"},
                status=200
            )

        # Generate API key that starts with lime_
        api_key = "lime_" + ''.join(random.choices(string.ascii_letters + string.digits, k=32))

        # Prepare user data to insert
        user_data = {
            "business_name": business_name,
            "password": password,
            "emails": emails,
            "created_at": datetime.datetime.today(),
            "plan": "free",
            "api_key": api_key,
            "mau_tracking": {},
        }

        # Insert the user into the Users collection
        result = users_collection.insert_one(user_data)

        if result.inserted_id:
            frontend = {
                "user_id": str(result.inserted_id),
                "body": "#90F08C",
                "eyes": "#FFFFFF",
                "size": 12,
                "context_params": [],
                "url": "",
                "folders": [],
                "api_key": api_key,
            }

            backend = {
                "user_id": str(result.inserted_id),
                "url": "",
                "folders": [],
                "api_key": api_key,
            }

            frontend_result = frontend_collection.insert_one(frontend)
            backend_result = backend_collection.insert_one(backend)

            users_collection.update_one({"_id": result.inserted_id}, {"$set": {"frontend": str(frontend_result.inserted_id), "backend": str(backend_result.inserted_id)}})

        if result.inserted_id:
            return JsonResponse(
                {"success": True, "user": str(result.inserted_id)},
                status=200
            )
        else:
            raise Exception("Failed to insert user")
    
    except Exception as error:
        print("Error adding user:", error)
        return JsonResponse(
            {"message": "Internal Server Error", "error": str(error)},
            status=500
        )

@csrf_exempt
def sign_in(request):
    try:
        # Parse the incoming JSON request body
        data = json.loads(request.body)
        email = data.get("email")
        business_name = data.get("business_name")
        password = data.get("password")

        # Validate the request data
        if not business_name or not password or not email:
            return JsonResponse(
                {"message": "Invalid payload: Missing business_name, password, or email"},
                status=400
            )

        # Check if the user exists and credentials match
        user = users_collection.find_one({
            "business_name": business_name,
            "password": password,
            "emails": {"$in": [email]}
        })

        if user:
            return JsonResponse(
                {"success": True, "user": str(user["_id"])},
                status=200
            )
        else:
            return JsonResponse(
                {"warning": "Invalid credentials"},
                status=200
            )

    except Exception as error:
        print("Error signing in:", error)
        return JsonResponse(
            {"message": "Internal Server Error", "error": str(error)},
            status=500
        )

@csrf_exempt
def user_details(request):
    try:
        # Parse the incoming JSON request body
        data = json.loads(request.body)
        user_id = data.get("user_id")

        # Validate the request data
        if not user_id:
            return JsonResponse(
                {"message": "Invalid payload: Missing user_id"},
                status=400
            )

        # Convert string ID to ObjectId for MongoDB query
        try:
            object_id = ObjectId(user_id)
        except:
            return JsonResponse(
                {"message": "Invalid user ID format"},
                status=400
            )

        # Find the user in the database
        user = users_collection.find_one({"_id": object_id})

        if user:
            # Convert ObjectId to string for JSON serialization
            user['_id'] = str(user['_id'])
            
            # Return user details (excluding password)
            return JsonResponse({
                "success": True,
                "user": {
                    "id": user['_id'],
                    "business_name": user['business_name'],
                    "emails": user['emails'],
                    "plan": user.get('plan', 'free'),
                    "created_at": user.get('created_at').isoformat() if user.get('created_at') else None,
                    "frontend": user.get('frontend'),
                    "backend": user.get('backend'),
                    "api_key": user.get('api_key'),
                    "last_paid": user.get('last_paid').isoformat() if user.get('last_paid') else None,
                }
            }, status=200)
        else:
            return JsonResponse(
                {"warning": "User not found"},
                status=404
            )

    except Exception as error:
        print("Error fetching user details:", error)
        return JsonResponse(
            {"message": "Internal Server Error", "error": str(error)},
            status=500
        )
    

@csrf_exempt
def update_emails(request):
    try:
        # Parse the incoming JSON request body
        data = json.loads(request.body)
        user_id = data.get("user_id")
        emails = data.get("emails")

        # Validate the request data
        if not user_id:
            return JsonResponse(
                {"message": "Invalid payload: Missing user_id"},
                status=400
            )
        
        if not emails or not isinstance(emails, list):
            return JsonResponse(
                {"message": "Invalid payload: emails must be a non-empty list"},
                status=400
            )

        # Convert string ID to ObjectId for MongoDB query
        try:
            object_id = ObjectId(user_id)
        except:
            return JsonResponse(
                {"message": "Invalid user ID format"},
                status=400
            )

        # Update the user's emails in the database
        result = users_collection.update_one(
            {"_id": object_id},
            {"$set": {"emails": emails}}
        )

        if result.matched_count > 0:
            if result.modified_count > 0:
                return JsonResponse({
                    "success": True,
                    "message": "Emails updated successfully"
                }, status=200)
            else:
                return JsonResponse({
                    "success": True,
                    "message": "No changes made to emails"
                }, status=200)
        else:
            return JsonResponse(
                {"warning": "User not found"},
                status=404
            )

    except Exception as error:
        print("Error updating user emails:", error)
        return JsonResponse(
            {"message": "Internal Server Error", "error": str(error)},
            status=500
        )


@csrf_exempt
def update_frontend(request):
    try:
        # Parse the incoming JSON request body
        data = json.loads(request.body)
        user_id = data.get("user_id")

        # Validate the request data
        if not user_id:
            return JsonResponse(
                {"message": "Invalid payload: Missing user_id"},
                status=400
            )

        # Convert string ID to ObjectId for MongoDB query
        try:
            object_id = ObjectId(user_id)
        except:
            return JsonResponse(
                {"message": "Invalid user ID format"},
                status=400
            )

        # Find the user in the database
        user = users_collection.find_one({"_id": object_id})
        
        if not user:
            return JsonResponse(
                {"warning": "User not found"},
                status=404
            )
            
        # Get the frontend document ID
        frontend_id = user.get("frontend")
        if not frontend_id:
            return JsonResponse(
                {"warning": "Frontend configuration not found for this user"},
                status=404
            )
            
        try:
            frontend_object_id = ObjectId(frontend_id)
        except:
            return JsonResponse(
                {"message": "Invalid frontend ID format"},
                status=400
            )
            
        # Prepare update data
        update_data = {}
        
        # Check for each parameter and add to update if present
        if "body" in data:
            update_data["body"] = data["body"]
            
        if "eyes" in data:
            update_data["eyes"] = data["eyes"]
            
        if "size" in data:
            update_data["size"] = data["size"]
            
        if "context_params" in data:
            update_data["context_params"] = data["context_params"]
            
        if "url" in data:
            update_data["url"] = data["url"]

        if "folders" in data:
            update_data["folders"] = data["folders"]
        
        # New chat UI fields
        if "aiText" in data:
            update_data["aiText"] = data["aiText"]
            
        if "userText" in data:
            update_data["userText"] = data["userText"]
            
        if "pageBackground" in data:
            update_data["pageBackground"] = data["pageBackground"]
            
        if "aiMessageBackground" in data:
            update_data["aiMessageBackground"] = data["aiMessageBackground"]
            
        if "userMessageBackground" in data:
            update_data["userMessageBackground"] = data["userMessageBackground"]
            
        if "banner" in data:
            update_data["banner"] = data["banner"]
            
        if "pageTitle" in data:
            update_data["pageTitle"] = data["pageTitle"]
            
        if "startText" in data:
            update_data["startText"] = data["startText"]
            
        # If nothing to update, return success
        if not update_data:
            return JsonResponse(
                {"success": True, "message": "No updates provided"},
                status=200
            )
            
        # Update the frontend document
        result = frontend_collection.update_one(
            {"_id": frontend_object_id},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            return JsonResponse(
                {"success": True, "message": "Frontend updated successfully"},
                status=200
            )
        else:
            return JsonResponse(
                {"success": True, "message": "No changes made to frontend"},
                status=200
            )
    
    except Exception as error:
        print("Error updating frontend:", error)
        return JsonResponse(
            {"message": "Internal Server Error", "error": str(error)},
            status=500
        )
    

@csrf_exempt
def frontend_details(request):
    try:
        # Parse the incoming JSON request body
        data = json.loads(request.body)
        user_id = data.get("user_id")

        # Validate the request data
        if not user_id:
            return JsonResponse(
                {"message": "Invalid payload: Missing user_id"},
                status=400
            )
        
        user_object_id = ObjectId(user_id)
        user = users_collection.find_one({"_id": user_object_id})
        if not user:
            return JsonResponse(
                {"warning": "User not found"},
                status=404
            )
        
        frontend_id = user.get("frontend")
        if not frontend_id:
            return JsonResponse(
                {"warning": "No frontend configuration associated with this user"},
                status=404
            )

        # Convert frontend_id to ObjectId for MongoDB query
        try:
            frontend_object_id = ObjectId(frontend_id)
        except:
            return JsonResponse(
                {"message": "Invalid frontend ID format"},
                status=400
            )
            
        # Get the frontend document
        frontend = frontend_collection.find_one({"_id": frontend_object_id})
        
        if frontend:
            # Convert ObjectId to string for JSON serialization
            frontend['_id'] = str(frontend['_id'])
            frontend['user_id'] = str(frontend['user_id'])
            
            return JsonResponse({
                "success": True,
                "frontend": {
                    "id": frontend['_id'],
                    "user_id": frontend['user_id'],
                    # Original fields
                    "body": frontend.get('body', "#90F08C"),
                    "eyes": frontend.get('eyes', "#FFFFFF"),
                    "size": frontend.get('size', 12),
                    "context_params": frontend.get('context_params', []),
                    "url": frontend.get('url', ""),
                    "folders": frontend.get('folders', []),
                    "api_key": frontend.get('api_key'),
                    # New chat UI fields
                    "aiText": frontend.get('aiText', "#000000"),
                    "userText": frontend.get('userText', "#000000"),
                    "pageBackground": frontend.get('pageBackground', "#FFFFFF"),
                    "aiMessageBackground": frontend.get('aiMessageBackground', "#F3F4F6"),
                    "userMessageBackground": frontend.get('userMessageBackground', "#E5E7EB"),
                    "banner": frontend.get('banner', "#90F08C"),
                    "pageTitle": frontend.get('pageTitle', "Chat Assistant"),
                    "startText": frontend.get('startText', "How can I help you today?"),
                }
            }, status=200)
        else:
            return JsonResponse(
                {"warning": "Frontend configuration not found"},
                status=404
            )

    except Exception as error:
        print("Error fetching frontend details:", error)
        return JsonResponse(
            {"message": "Internal Server Error", "error": str(error)},
            status=500
        )
    

@csrf_exempt
def update_backend(request):
    try:
        # Parse the incoming JSON request body
        data = json.loads(request.body)
        user_id = data.get("user_id")

        # Validate the request data
        if not user_id:
            return JsonResponse(
                {"message": "Invalid payload: Missing user_id"},
                status=400
            )

        # Convert string ID to ObjectId for MongoDB query
        try:
            object_id = ObjectId(user_id)
        except:
            return JsonResponse(
                {"message": "Invalid user ID format"},
                status=400
            )

        # Find the user in the database
        user = users_collection.find_one({"_id": object_id})
        
        if not user:
            return JsonResponse(
                {"warning": "User not found"},
                status=404
            )
            
        # Get the backend document ID
        backend_id = user.get("backend")
        if not backend_id:
            return JsonResponse(
                {"warning": "Backend configuration not found for this user"},
                status=404
            )
            
        try:
            backend_object_id = ObjectId(backend_id)
        except:
            return JsonResponse(
                {"message": "Invalid backend ID format"},
                status=400
            )
            
        # Prepare update data
        update_data = {}
        
        # Check for URL and folders parameters
        if "url" in data:
            update_data["url"] = data["url"]

        if "folders" in data:
            update_data["folders"] = data["folders"]
            
        # If nothing to update, return success
        if not update_data:
            return JsonResponse(
                {"success": True, "message": "No updates provided"},
                status=200
            )
            
        # Update the backend document
        result = backend_collection.update_one(
            {"_id": backend_object_id},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            return JsonResponse(
                {"success": True, "message": "Backend updated successfully"},
                status=200
            )
        else:
            return JsonResponse(
                {"success": True, "message": "No changes made to backend"},
                status=200
            )
    
    except Exception as error:
        print("Error updating backend:", error)
        return JsonResponse(
            {"message": "Internal Server Error", "error": str(error)},
            status=500
        )
    

@csrf_exempt
def backend_details(request):
    try:
        # Parse the incoming JSON request body
        data = json.loads(request.body)
        user_id = data.get("user_id")

        # Validate the request data
        if not user_id:
            return JsonResponse(
                {"message": "Invalid payload: Missing user_id"},
                status=400
            )
        
        user_object_id = ObjectId(user_id)
        user = users_collection.find_one({"_id": user_object_id})
        if not user:
            return JsonResponse(
                {"warning": "User not found"},
                status=404
            )
        
        backend_id = user.get("backend")
        if not backend_id:
            return JsonResponse(
                {"warning": "No backend configuration associated with this user"},
                status=404
            )

        # Convert backend_id to ObjectId for MongoDB query
        try:
            backend_object_id = ObjectId(backend_id)
        except:
            return JsonResponse(
                {"message": "Invalid backend ID format"},
                status=400
            )
            
        # Get the backend document
        backend = backend_collection.find_one({"_id": backend_object_id})
        
        if backend:
            # Convert ObjectId to string for JSON serialization
            backend['_id'] = str(backend['_id'])
            backend['user_id'] = str(backend['user_id'])
            
            return JsonResponse({
                "success": True,
                "backend": {
                    "id": backend['_id'],
                    "user_id": backend['user_id'],
                    "url": backend.get('url', ""),
                    "folders": backend.get('folders', []),
                    "api_key": backend.get('api_key'),
                }
            }, status=200)
        else:
            return JsonResponse(
                {"warning": "Backend configuration not found"},
                status=404
            )

    except Exception as error:
        print("Error fetching backend details:", error)
        return JsonResponse(
            {"message": "Internal Server Error", "error": str(error)},
            status=500
        )
    

@csrf_exempt
def public_frontend_details(request):
    try:
        # Parse the incoming JSON request body
        data = json.loads(request.body)
        api_key = data.get("api_key")

        # Validate the request data
        if not api_key:
            return JsonResponse(
                {"message": "Invalid payload: Missing api_key"},
                status=400
            )
            
        # Get the frontend document
        frontend = frontend_collection.find_one({"api_key": api_key})
        
        if frontend:
            # Convert ObjectId to string for JSON serialization
            frontend['_id'] = str(frontend['_id'])
            frontend['user_id'] = str(frontend['user_id'])
            
            return JsonResponse({
                "success": True,
                "frontend": {
                    "id": frontend['_id'],
                    "user_id": frontend['user_id'],
                    "body": frontend.get('body', "#90F08C"),
                    "eyes": frontend.get('eyes', "#FFFFFF"),
                    "size": frontend.get('size', 12),
                    "context_params": frontend.get('context_params', []),
                    "url": frontend.get('url', ""),
                    "folders": frontend.get('folders', []),
                    "api_key": frontend.get('api_key'),
                    # New chat UI fields
                    "aiText": frontend.get('aiText', "#000000"),
                    "userText": frontend.get('userText', "#000000"),
                    "pageBackground": frontend.get('pageBackground', "#FFFFFF"),
                    "aiMessageBackground": frontend.get('aiMessageBackground', "#F3F4F6"),
                    "userMessageBackground": frontend.get('userMessageBackground', "#E5E7EB"),
                    "banner": frontend.get('banner', "#90F08C"),
                    "pageTitle": frontend.get('pageTitle', "Chat Assistant"),
                    "startText": frontend.get('startText', "How can I help you today?"),
                }
            }, status=200)
        else:
            return JsonResponse(
                {"warning": "Frontend configuration not found"},
                status=404
            )

    except Exception as error:
        print("Error fetching frontend details:", error)
        return JsonResponse(
            {"message": "Internal Server Error", "error": str(error)},
            status=500
        )
    
    

import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek

# logging.basicConfig(level=logging.DEBUG)

class EndpointSelection(BaseModel):
    endpoint_name: str = Field(description="Name of the selected endpoint")
    reason: str = Field(description="Reason for selecting this endpoint")
    suitable: bool = Field(description="Whether any endpoint is suitable for the request")

class EndpointAgent:
    def __init__(self, users_collection, backend_collection, frontend_collection, deepseek_api_key=None):
        self.users_collection = users_collection
        self.backend_collection = backend_collection
        self.frontend_collection = frontend_collection
        
        # Initialize LangChain components if API key is provided
        self.has_llm = deepseek_api_key is not None
        if self.has_llm:
            self.llm = ChatDeepSeek(
                model="deepseek-chat",   
                temperature=0,                      
                api_key=deepseek_api_key,
            )
            
            # Create parsers
            self.endpoint_parser = PydanticOutputParser(pydantic_object=EndpointSelection)
            # Add a fixing layer to handle parsing errors
            self.endpoint_parser = OutputFixingParser.from_llm(parser=self.endpoint_parser, llm=self.llm)
            
            # Create prompt templates
            self.endpoint_selection_prompt = PromptTemplate(
                template="""
                I want to {user_prompt} by hitting one the following endpoints. 
                
                Available Endpoints:
                {endpoints_json}
                
                If an endpoint matches my request, tell me which endpoint I should hit, if any at all (don't give an endpoint to hit if you aren't sure it fits my needs). 
                
                {format_instructions}
                
                If none of the endpoints can fulfill the request, set suitable to false.
                """,
                input_variables=["endpoints_json", "user_prompt"],
                partial_variables={"format_instructions": self.endpoint_parser.get_format_instructions()}
            )
            
            self.schema_filling_prompt = PromptTemplate(
                template="""
                Example Schema:
                {schema_json}  
               
                I want you to {user_prompt}, so change the schema json to fit what I am trying to do. Here is some context: {user_context}
                Return a valid JSON object and nothing else. This schema is going to be sent to an endpoint called {endpoint_name} and it's described as {endpoint_description}
                Fill in values surrounded by curly braces in schema like "user_id": "{{user_id}}" with things in the
                context I gave you. 

                Don't modify the structure of the schema, just change the values so it matches what I'm trying to do. 

                If you don't think you can do this because you need more context from me or more instructions to fill the schema, please just message me back "I need this ---"
                """,
                input_variables=["endpoint_name", "endpoint_description", "schema_json", "user_prompt", "user_context"]
            )

            self.response_creation_prompt = PromptTemplate(
                template="""
                Prompt: {user_prompt}
                Request processed using endpoint: {endpoint_name} that is described as {endpoint_description}

                Response from server: {server_response}

                Give a message that sums this up without any URL links just the URL text itself if needed and make it short and to the point and 
                have nothing about the endpoint used or its name or what was done to process the user's request. just give the useful info the user needs.
                """,
                input_variables=["endpoint_name", "endpoint_description", "server_response", "user_prompt"]
            )

            self.no_endpoint_response_creation_prompt = PromptTemplate(
                template="""
                Prompt: {user_prompt}

                Give a message that responds to this given as a member of company {company_name}. If you don't know what to respond give a message that 
                tells the user to check the docs or contact support or etc depending on the prompt. Make the message short and to the point. 
                """,
                input_variables=["company_name", "user_prompt"]
            )
    
    def process_prompt(self, prompt: str, api_key: str, context: Optional[Dict] = None):
        if context is None:
            context = {}

        logging.debug(f"Received request with prompt: {prompt}")
        logging.debug(f"API Key received: {api_key}")
        logging.debug(f"Context received: {context}")

        # Step 1: Verify user
        user = self.users_collection.find_one({"api_key": api_key})
        if not user:
            logging.error("User not found.")
            return {"error": "User not found", "status": 404}

        logging.debug(f"User found: {user}")

        # Step 2: Get backend and frontend document IDs
        backend_id = user.get("backend")
        frontend_id = user.get("frontend")

        logging.debug(f"Backend ID: {backend_id}, Frontend ID: {frontend_id}")

        if not backend_id and not frontend_id:
            logging.error("No configuration found for this user.")
            return {"error": "No configuration found for this user", "status": 404}

        # Step 3: Retrieve backend and frontend configurations
        backend_config = self.backend_collection.find_one({"_id": ObjectId(backend_id)}) if backend_id else None
        frontend_config = self.frontend_collection.find_one({"_id": ObjectId(frontend_id)}) if frontend_id else None

        logging.debug(f"Backend Config: {json.dumps(backend_config, default=str) if backend_config else 'None'}")
        logging.debug(f"Frontend Config: {json.dumps(frontend_config, default=str) if frontend_config else 'None'}")

        if not backend_config and not frontend_config:
            logging.error("Configuration documents not found.")
            return {"error": "Configuration documents not found", "status": 404}

        # Step 4: Get all endpoints
        backend_endpoints = self._get_all_endpoints(backend_config) if backend_config else []
        frontend_endpoints = self._get_all_frontend_endpoints(frontend_config) if frontend_config else []

        logging.debug(f"Retrieved {len(backend_endpoints)} backend endpoints and {len(frontend_endpoints)} frontend endpoints.")

        all_endpoints = []
        for endpoint in backend_endpoints:
            endpoint["endpoint_type"] = "backend"
            all_endpoints.append(endpoint)

        for endpoint in frontend_endpoints:
            endpoint["endpoint_type"] = "frontend"
            all_endpoints.append(endpoint)  

        if not all_endpoints:
            logging.error("No endpoints available in configuration.")
            return {"formatted_response": "No endpoints available in your configuration", "status": 200}

        logging.debug(f"Total available endpoints: {json.dumps(all_endpoints, indent=2)}")

        # Step 5: Find the best matching endpoint
        best_endpoint = self._find_best_endpoint(prompt, all_endpoints)

        if not best_endpoint:
            logging.warning("No suitable endpoint found.")

            formatted_response = self.llm.invoke(self.no_endpoint_response_creation_prompt.format(
                company_name=user.get("business_name"),
                user_prompt=prompt,
            ))

            return {"formatted_response": str(formatted_response.content), "status": 200}

        logging.debug(f"Selected Endpoint: {json.dumps(best_endpoint, indent=2)}")

        # Step 6: Handle the selected endpoint
        if best_endpoint.get("endpoint_type") == "frontend":
            logging.info("Returning frontend endpoint URL.")
            # print("BEST_ENDPOINT_NAME: " + best_endpoint.get("name", ""))
            # print("BEST_ENDPOINT_DESC: " + best_endpoint.get("description", ""))
            # print("USER_PROMPT: " + prompt)
            # print("SERVER_RESPONSE: " + best_endpoint.get("url"))

            # print("BEST_ENDPOINT_ID: " + best_endpoint.get("id", ""))
            # print("BEST_ENDPOINT_FOLDER_ID: " + best_endpoint.get("folder_id", ""))

            frontend_collection.update_one(
                {"_id": ObjectId(frontend_id), "folders.id": best_endpoint.get("folder_id", ""), "folders.endpoints.id": best_endpoint.get("id", "")},
                {"$inc": {"folders.$[folder].endpoints.$[endpoint].num_hits": 1}},
                array_filters=[
                    {"folder.id": best_endpoint.get("folder_id", "")},
                    {"endpoint.id": best_endpoint.get("id", "")}
                ]
             )

            formatted_response = self.llm.invoke(self.response_creation_prompt.format(
                endpoint_name=best_endpoint.get("name", ""),
                endpoint_description=best_endpoint.get("description", ""),
                user_prompt=prompt,
                server_response=best_endpoint.get("url", "")
            ))

            return {
                "status": 200,
                "endpoint_type": "frontend",
                "url": best_endpoint.get("url", ""),
                "description": best_endpoint.get("description", ""), 
                "formatted_response": str(formatted_response.content),
            }
        
        else:
            # Step 7: Fill schema and send request for backend endpoint
            logging.info("Processing backend endpoint request.")
            filled_schema = self._fill_schema(prompt, best_endpoint, context)
    
            # Add check for clarification needs
            if isinstance(filled_schema, str):
                message = filled_schema
                return {"message": message, "status": 200, "needs_clarification": True}
            
            if "error" in filled_schema:
                logging.error(f"Schema filling error: {filled_schema}")
                return {"message": filled_schema.get("error"), "status": 200}

            logging.debug(f"Filled schema: {json.dumps(filled_schema, indent=2)}")

            # Step 8: Send request
            response = self._send_request(best_endpoint, filled_schema)
            logging.debug(f"Response from backend: {response}")

            backend_collection.update_one(
                {"_id": ObjectId(backend_id), "folders.id": best_endpoint.get("folder_id", ""), "folders.endpoints.id": best_endpoint.get("id", "")},
                {"$inc": {"folders.$[folder].endpoints.$[endpoint].num_hits": 1}},
                array_filters=[
                    {"folder.id": best_endpoint.get("folder_id", "")},
                    {"endpoint.id": best_endpoint.get("id", "")}
                ]
             )  

            formatted_response = self.llm.invoke(self.response_creation_prompt.format(
                endpoint_name=best_endpoint.get("name", ""),
                endpoint_description=best_endpoint.get("description", ""),
                user_prompt=prompt,
                server_response=response
            ))

            return {
                "status": 200,
                "endpoint_type": "backend",
                "endpoint_used": best_endpoint.get("name", "Unknown endpoint"),
                "request_data": filled_schema,
                "response": response, 
                "formatted_response": str(formatted_response.content),
            }
        
    
    def _get_all_endpoints(self, backend_config: Dict) -> List[Dict]:
        """
        Extract all endpoints from the backend configuration
        """
        if not backend_config:
            return []
            
        endpoints = []
        folders = backend_config.get("folders", [])
        
        # Process each folder
        for folder in folders:
            folder_endpoints = folder.get("endpoints", [])
            folder_id = folder.get("id", "")
            for endpoint in folder_endpoints:
                if "name" in endpoint and "url" in endpoint and "schema" in endpoint:
                    endpoint['folder_id'] = folder_id
                    endpoints.append(endpoint)
        
        return endpoints
    
    def _get_all_frontend_endpoints(self, frontend_config: Dict) -> List[Dict]:
        """
        Extract all endpoints from the frontend configuration
        """
        if not frontend_config:
            return []
            
        endpoints = []
        folders = frontend_config.get("folders", [])
        
        # Process each folder
        for folder in folders:
            folder_endpoints = folder.get("endpoints", [])
            folder_id = folder.get("id", "")
            for endpoint in folder_endpoints:
                if "name" in endpoint and "url" in endpoint:
                    endpoint['folder_id'] = folder_id
                    endpoints.append(endpoint)
        
        return endpoints
    
    def _find_best_endpoint(self, prompt: str, endpoints: List[Dict]) -> Optional[Dict]:
        """
        Find the best matching endpoint based on the user prompt
        """
        try:
            if self.has_llm:
                result = self._find_endpoint_with_langchain(prompt, endpoints)
                if result:
                    return result
        except Exception as e:
            print("Find Best Endpoint: " + traceback.format_exc())
            print(f"LLM endpoint selection failed: {str(e)}")
        
        # Fall back to keyword matching
        return self._find_endpoint_with_keywords(prompt, endpoints)
    
    
    def _find_endpoint_with_keywords(self, prompt: str, endpoints: List[Dict]) -> Optional[Dict]:
        """
        Fallback method using keyword matching when LLM is not available.
        """
        prompt_lower = prompt.lower()
        best_match = None
        highest_score = 0

        logging.debug(f"Matching prompt: {prompt_lower}")

        for endpoint in endpoints:
            name = endpoint.get("name", "").lower()
            description = endpoint.get("description", "").lower()
            
            score = 0
            if name in prompt_lower:
                score += 5
            
            for word in name.split():
                if len(word) > 3 and word in prompt_lower:
                    score += 1

            for word in description.split():
                if len(word) > 3 and word in prompt_lower:
                    score += 0.5

            logging.debug(f"Endpoint '{endpoint.get('name', '')}' scored {score}")

            if score > highest_score:
                highest_score = score
                best_match = endpoint

        # Only return if the score is above a threshold
        if highest_score > 0:
            logging.info(f"Best matched endpoint: {best_match.get('name', 'Unknown')}")
            return best_match
        
        logging.warning("No suitable endpoint found using keyword matching.")
        return None
    

    def _find_endpoint_with_langchain(self, prompt: str, endpoints: List[Dict]) -> Optional[Dict]:
        """
        Use LangChain and DeepSeek to find the best endpoint based on understanding the prompt
        """
        # Format endpoints for the LLM, including endpoint type
        endpoints_json = json.dumps([{
            "name": ep.get("name", ""),
            "description": ep.get("description", ""),
            "url": ep.get("url", ""),
            "type": ep.get("endpoint_type", "backend")  # Include the type
        } for ep in endpoints], indent=2)
        
        # Create a chain for endpoint selection
        try:
            result = self.llm.invoke(self.endpoint_selection_prompt.format(
                endpoints_json=endpoints_json, 
                user_prompt=prompt
            ))
            
            # Parse the result
            try:
                endpoint_selection = self.endpoint_parser.parse(result.content)
                
                # Only proceed if a suitable endpoint was found
                if endpoint_selection.suitable:
                    # Find the endpoint with matching name
                    for endpoint in endpoints:
                        if endpoint.get("name", "") == endpoint_selection.endpoint_name:
                            return endpoint
            except Exception as parse_error:
                logging.error(f"Error parsing endpoint selection: {str(parse_error)}")
                return None
                
        except Exception as e:
            logging.error(f"Error in endpoint selection: {str(e)}")
            if "Insufficient Balance" in str(e) or "402" in str(e):
                logging.warning("Detected insufficient balance, falling back to keyword matching")
            return None
        
        return None
    
    
    def replace_context_placeholders(self, data, context: Dict):
        """
        Recursively replace {key} placeholders in strings with values from context.
        """
        if isinstance(data, dict):
            return {k: self.replace_context_placeholders(v, context) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.replace_context_placeholders(item, context) for item in data]
        elif isinstance(data, str):
            # Use regex to find and replace {key} patterns
            def replacer(match):
                key = match.group(1)
                return str(context.get(key, f'{{{key}}}'))  # Replace or retain placeholder
            return re.sub(r'\{(\w+)\}', replacer, data)
        else:
            return data

    def _fill_schema(self, prompt: str, endpoint: Dict, context: Dict) -> Dict:
        """
        Fill in the endpoint schema based on the user prompt.
        """
        if endpoint.get("endpoint_type") == "frontend":
            return {}

        try:
            schema = endpoint.get("schema", "{}")
            schema_dict = json.loads(schema) if isinstance(schema, str) else schema
            if not schema_dict:
                return {}

            # Replace placeholders in the original schema with context values
            schema_dict = self.replace_context_placeholders(schema_dict, context)

            if self.has_llm:
                return self._fill_schema_with_langchain(prompt, endpoint, schema_dict, context)
            else:
                return self._simple_fill_from_example(prompt, schema_dict, context)
        except json.JSONDecodeError:
            logging.error(f"Invalid schema JSON: {schema}")
            return {"error": "Invalid schema format"}
        except Exception as e:
            logging.error(f"Error filling schema: {str(e)}")
            return {"error": f"Failed to fill schema: {str(e)}"}

    def _fill_schema_with_langchain(self, prompt: str, endpoint: Dict, schema_dict: Dict, context: Dict) -> Dict:
        """
        Use LangChain to fill in the schema based on the user prompt.
        """
        # Generate schema JSON with placeholders already replaced
        schema_json = json.dumps(schema_dict, indent=2)
        context_json = json.dumps(context, indent=2)
        print("ENDPOINT_NAME:" + endpoint.get("name", ""))
        print("SCHEMA_JSON:" + schema_json)
        print("USER_PROMPT:" + prompt)
        print("CONTEXT JSON:" + context_json)

        try:
            result = self.llm.invoke(self.schema_filling_prompt.format(
                endpoint_name=endpoint.get("name", ""),
                endpoint_description=endpoint.get("description", ""),
                schema_json=schema_json,
                user_prompt=prompt,
                user_context=context_json
            ))

            if isinstance(result, str):
                return result

            # Extract JSON from response
            try:
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', result.content, re.DOTALL)
                filled_schema = json.loads(json_match.group(1)) if json_match else json.loads(result.content)
                
                # Ensure any remaining placeholders are replaced
                filled_schema = self.replace_context_placeholders(filled_schema, context)
                return filled_schema
            except json.JSONDecodeError:
                logging.warning("Failed to parse LLM response as JSON, using example schema")
                return schema_dict

        except Exception as e:
            logging.error(f"Error in schema filling: {str(e)}")
            return schema_dict
            
    
    def _simple_fill_from_example(self, prompt: str, schema_dict: Dict) -> Dict:
        """
        Use the schema as-is since it's already an example
        """
        return schema_dict
    
    def _send_request(self, endpoint: Dict, data: Dict) -> Dict:
        # Only execute for backend endpoints
        if endpoint.get("endpoint_type") == "frontend":
            return {"error": "Cannot execute frontend endpoints"}
            
        url = endpoint.get("url", "")
        logging.info(f"Sending request to: {url}")
        method = endpoint.get("method", "POST").upper()

        logging.debug(f"Sending {method} request with data: {json.dumps(data)}")

        try:
            if method == "GET":
                response = requests.get(url, params=data)
            elif method == "POST":
                response = requests.post(url, json=data)
            elif method == "PUT":
                response = requests.put(url, json=data)
            elif method == "DELETE":
                response = requests.delete(url, json=data)
            else:
                return {"error": f"Unsupported HTTP method: {method}"}

            logging.debug(f"Response Status: {response.status_code}")
            logging.debug(f"Response Data: {response.text[:500]}...")  # Log truncated response

            if 200 <= response.status_code < 300:
                try:
                    return response.json()
                except json.JSONDecodeError:
                    return {"text": response.text}
            else:
                return {
                    "error": f"Request failed with status code: {response.status_code}",
                    "details": response.text[:1000]  # Truncate long error responses
                }
        except Exception as e:
            logging.error("Send Request Error: " + traceback.format_exc())
            return {"error": f"Request failed: {str(e)}"}


@csrf_exempt
def process_prompt(request):
    try:
        data = json.loads(request.body)
        prompt = data.get("prompt")
        api_key = data.get("api_key")
        context = data.get("context", {})
        
        # Extract client_info from context if it exists
        client_info = context.get("client_info", {})
        if not client_info and "client_info" in data:
            client_info = data.get("client_info", {})
        
        if not prompt or not api_key:
            return JsonResponse(
                {"message": "Missing prompt or API key"},
                status=400
            )
        
        # Track MAU if client_info is available
        if client_info:
            is_new_mau = update_mau(api_key, client_info)
            logging.info(f"MAU tracking: client is {'new' if is_new_mau else 'returning'} for this month")
        
        # Initialize the agent
        agent = EndpointAgent(
            users_collection=users_collection,
            backend_collection=backend_collection,
            frontend_collection=frontend_collection,
            deepseek_api_key=f'{settings.DEEPSEEK_API_KEY}',
        )
        
        # Process the prompt
        result = agent.process_prompt(prompt, api_key, context)
        
        # Ensure we return 200 status even if no endpoint is found
        if "status" in result and result["status"] != 200:
            result_status = result.pop("status", 200)
            return JsonResponse(result, status=result_status)
        else:
            return JsonResponse(result, status=200)
    except Exception as error:
        logging.error("process_prompt error: " + traceback.format_exc())
        return JsonResponse(
            {"message": "Internal Server Error", "error": str(error)},
            status=500
        )
    


def update_mau(api_key, client_info):
    """
    Updates the Monthly Active Users (MAU).
    
    Args:
        users_collection: MongoDB collection containing user data
        api_key: API key
        client_info: Dictionary containing client identification information
    
    Returns:
        bool: True if the client is a new MAU for the current month, False otherwise
    """
    try:
        # Get current month and year for MAU tracking
        current_month = datetime.datetime.now().strftime("%Y-%m")
        
        # Extract client fingerprint or create one from available data
        fingerprint = client_info.get("fingerprint")
        if not fingerprint and client_info:
            # Create a fingerprint from available client data if not provided
            components = [
                client_info.get("user_agent", ""),
                client_info.get("language", ""),
                client_info.get("screen_resolution", ""),
                client_info.get("timezone", ""),
                client_info.get("hostname", ""),
                client_info.get("referrer", "")
            ]
            fingerprint = hashlib.sha256('|'.join(components).encode()).hexdigest()
        
        if not fingerprint:
            logging.warning("No client fingerprint available for MAU tracking")
            return False
            
        # Find the user document by API key
        user = users_collection.find_one({"api_key": api_key})
        if not user:
            logging.warning(f"No user found with API key: {api_key}")
            return False
        
        # Initialize MAU tracking structure if it doesn't exist
        if "mau_tracking" not in user:
            users_collection.update_one(
                {"api_key": api_key},
                {"$set": {"mau_tracking": {}}}
            )
            user["mau_tracking"] = {}
            
        # Initialize current month's tracking if it doesn't exist
        if current_month not in user["mau_tracking"]:
            users_collection.update_one(
                {"api_key": api_key},
                {"$set": {f"mau_tracking.{current_month}": []}}
            )
            user["mau_tracking"][current_month] = []
        
        # Check if this client is already counted in the current month
        if fingerprint in user["mau_tracking"][current_month]:
            return False  # Not a new MAU
        
        # Add the client to the current month's MAU list
        users_collection.update_one(
            {"api_key": api_key},
            {"$push": {f"mau_tracking.{current_month}": fingerprint}}
        )
        
        # Increment the MAU counter
        # Maintain a counter separate from the array for quick access
        current_count = user.get("mau_count", {}).get(current_month, 0)
        users_collection.update_one(
            {"api_key": api_key},
            {"$set": {f"mau_count.{current_month}": current_count + 1}}
        )
        
        return True  # New MAU added
        
    except Exception as e:
        logging.error(f"Error updating MAU tracking: {traceback.format_exc()}")
        return False
    


@csrf_exempt
def get_mau_stats(request):
    """
    Get MAU statistics for a specific block over a number of months.
    
    Args:
        users_collection: MongoDB collection containing user data
        api_key: API key
        months: Number of months to include in the statistics (default 6)
    
    Returns:
        dict: Monthly MAU statistics
    """
    try:
        data = json.loads(request.body)
        user_id = data.get("user_id")
        months = data.get("months", 6)

        user = users_collection.find_one({"_id": ObjectId(user_id)})
        if not user or "mau_tracking" not in user:
            return JsonResponse({"error": "No MAU data found"}, status=404)
        
        # Get the current month and previous months
        current_date = datetime.datetime.now()
        month_stats = {}

        for i in range(months):
            month_date = current_date.replace(day=1) - relativedelta(months=i)
            month_key = month_date.strftime("%Y-%m")
            
            # Get MAU count for this month
            month_stats[month_key] = user.get("mau_count", {}).get(month_key, 0)

        return JsonResponse(
            {"success": True, "mau_stats": month_stats},
            status=200
        )

    except Exception as error:
        return JsonResponse(
            {"message": "Internal Server Error", "error": str(error)},
            status=500
        )


# STRIPE CHECKOUT STUFF

# Set Stripe API key
stripe.api_key = settings.STRIPE_SK

# Stripe webhook secret
WEBHOOK_SECRET = settings.STRIPE_WEBHOOK_SECRET

@csrf_exempt
def create_checkout_session(request):
    """
    Creates a Stripe Checkout Session for recurring monthly payments.
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            product_id = data.get("product_id")
            user_id = data.get("user_id")
            print(user_id)

            if not user_id:
                return JsonResponse({"url": "http://localhost:3000/"})

            # Product price mapping for recurring subscriptions
            product_to_price_mapping = {
                "prod_RvtLgN1zOEG6iD": "price_1R21YdEK41Y7N6vui07ggvAY",
                "prod_RvtL5HMB98eLLm": "price_1R21Z5EK41Y7N6vuuu4c44kW"
            }

            if product_id not in product_to_price_mapping:
                return JsonResponse({"error": "Invalid Product ID"}, status=400)

            # Create a Stripe Checkout Session for recurring payments
            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=[
                    {
                        "price": product_to_price_mapping[product_id],
                        "quantity": 1,
                    }
                ],
                mode="subscription",  # Recurring subscription mode
                success_url="http://localhost:3000/dashboard",
                cancel_url="http://localhost:3000/dashboard",
                metadata={
                    "user_id": user_id,  # Attach user ID as metadata
                    "product_id": product_id,  # Attach product ID as metadata
                },
                subscription_data={
                    "metadata": {
                        "user_id": user_id,
                        "product_id": product_id,
                    }
                }
            )

            return JsonResponse({"url": session.url})

        except Exception as e:
            print(traceback.format_exc())
            return JsonResponse({"error": str(e)}, status=400)

@csrf_exempt
def stripe_webhook(request):
    """
    Handles Stripe webhook events.
    """
    payload = request.body
    sig_header = request.META['HTTP_STRIPE_SIGNATURE']

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, WEBHOOK_SECRET
        )
    except stripe.error.SignatureVerificationError as e:
        # Signature doesn't match
        return JsonResponse({'error': 'Invalid signature'}, status=400)

    # Handle checkout.session.completed
    if event["type"] == "checkout.session.completed" or event["type"] == "invoice.paid":
        session = event["data"]["object"]
        handle_checkout_session(session)

    return JsonResponse({"status": "success"}, status=200)


def handle_checkout_session(session):
    """
    Processes the checkout session or invoice payment.
    """
    # Check if this is a checkout session or invoice
    if 'metadata' in session and session.get('metadata'):
        # This is a checkout.session.completed event
        user_id = session["metadata"].get("user_id")
        product_id = session["metadata"].get("product_id")
    elif session.get('subscription'):
        # This is an invoice.paid event
        try:
            # Get the subscription to access metadata
            subscription = stripe.Subscription.retrieve(session.get('subscription'))
            # Get metadata from the subscription
            user_id = subscription.metadata.get('user_id')
            product_id = subscription.metadata.get('product_id')
        except Exception as e:
            print(f"Error retrieving subscription data: {e}")
            return
    else:
        print("Missing required data in session")
        return
        
    print("handling payment...")

    if not user_id:
        print("no user id")
        return
        
    user = users_collection.find_one({'_id': ObjectId(user_id)})
    
    if not user:
        print(f"User not found: {user_id}")
        return

    if product_id == "prod_RvtLgN1zOEG6iD":
        try:
            users_collection.update_one({'_id': ObjectId(user_id)}, {
                '$set': {'plan': 'startup', 'last_paid': datetime.datetime.today()}
            })
            print(f"Added/Updated Startup Plan to user {user_id}.")
        except Exception as e:
            print(f"Failed to update MongoDB: {e}")
    
    elif product_id == "prod_RvtL5HMB98eLLm":
        try:
            users_collection.update_one({'_id': ObjectId(user_id)}, {
                '$set': {'plan': 'enterprise', 'last_paid': datetime.datetime.today()}
            })
            print(f"Added/Updated Enterprise Plan to user {user_id}.")
        except Exception as e:
            print(f"Failed to update MongoDB: {e}")
    else:
        print(f"Unknown product ID: {product_id}")