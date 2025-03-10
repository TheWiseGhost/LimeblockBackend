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

from typing import Dict, List, Optional, Any
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.output_parsers import OutputFixingParser

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
                    "body": frontend.get('body', "#90F08C"),
                    "eyes": frontend.get('eyes', "#FFFFFF"),
                    "size": frontend.get('size', 12),
                    "context_params": frontend.get('context_params', []),
                    "url": frontend.get('url', ""),
                    "folders": frontend.get('folders', []),
                    "api_key": frontend.get('api_key'),
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
                max_tokens=1000,          
                api_key=f'{settings.DEEPSEEK_API_KEY}',
            )
            
            # Create parsers
            self.endpoint_parser = PydanticOutputParser(pydantic_object=EndpointSelection)
            # Add a fixing layer to handle parsing errors
            self.endpoint_parser = OutputFixingParser.from_llm(parser=self.endpoint_parser, llm=self.llm)
            
            # Create prompt templates
            self.endpoint_selection_prompt = PromptTemplate(
                template="""
                You are an API endpoint selection assistant. Given a user request and available endpoints, 
                select the most appropriate endpoint to fulfill the request.
                
                Available Endpoints:
                {endpoints_json}
                
                User Request: {user_prompt}
                
                {format_instructions}
                
                If none of the endpoints can fulfill the request, set suitable to false.
                """,
                input_variables=["endpoints_json", "user_prompt"],
                partial_variables={"format_instructions": self.endpoint_parser.get_format_instructions()}
            )
            
            self.schema_filling_prompt = PromptTemplate(
                template="""
                You are an API parameter extraction assistant. Given a user request and endpoint schema, 
                extract the appropriate values for each parameter.
                
                Endpoint: {endpoint_name}
                
                Schema:
                {schema_json}
                
                Example (if available):
                {example_json}
                
                User Request: {user_prompt}
                
                Extract values for each parameter in the schema based on the user request.
                Return a valid JSON object that contains parameter names as keys and extracted values that match the expected types.
                Use the example values as defaults if you cannot extract from the user request.
                """,
                input_variables=["endpoint_name", "schema_json", "example_json", "user_prompt"]
            )
    
    def process_prompt(self, prompt: str, api_key: str):
        """
        Main method to process a user prompt using their API key
        """
        # Step 1: Verify user and get their backend configuration
        user = self.users_collection.find_one({"api_key": api_key})
        if not user:
            return {"error": "User not found", "status": 404}
        
        # Get the backend and frontend document IDs
        backend_id = user.get("backend")
        frontend_id = user.get("frontend")
        
        if not backend_id and not frontend_id:
            return {"error": "No configuration found for this user", "status": 404}
        
        # Step 2: Retrieve the backend and frontend configurations
        backend_config = None
        frontend_config = None
        
        if backend_id:
            backend_config = self.backend_collection.find_one({"_id": ObjectId(backend_id)})
        
        if frontend_id:
            frontend_config = self.frontend_collection.find_one({"_id": ObjectId(frontend_id)})
        
        if not backend_config and not frontend_config:
            return {"error": "Configuration documents not found", "status": 404}
        
        # Step 3: Get all endpoints from both backend and frontend
        backend_endpoints = self._get_all_endpoints(backend_config) if backend_config else []
        frontend_endpoints = self._get_all_frontend_endpoints(frontend_config) if frontend_config else []
        
        # Combine all endpoints with a type flag to distinguish them
        all_endpoints = []
        for endpoint in backend_endpoints:
            endpoint["endpoint_type"] = "backend"
            all_endpoints.append(endpoint)
        
        for endpoint in frontend_endpoints:
            endpoint["endpoint_type"] = "frontend"
            all_endpoints.append(endpoint)
        
        if not all_endpoints:
            return {"error": "No endpoints available in your configuration", "status": 400}
        
        # Step 4: Find the best matching endpoint
        best_endpoint = self._find_best_endpoint(prompt, all_endpoints)
        if not best_endpoint:
            return {"error": "I don't think I can do this with your available endpoints", "status": 400}
        
        # Step 5: Handle the endpoint based on its type
        if best_endpoint.get("endpoint_type") == "frontend":
            # For frontend endpoints, just return the URL
            return {
                "url": best_endpoint.get("url", ""),
                "description": best_endpoint.get("description", "")
            }
        else:
            # For backend endpoints, fill the schema and send the request
            filled_schema = self._fill_schema(prompt, best_endpoint)
            if "error" in filled_schema:
                return filled_schema
            
            # Send the request to the endpoint
            response = self._send_request(best_endpoint, filled_schema)
            
            return {
                "status": 200,
                "endpoint_type": "backend",
                "endpoint_used": best_endpoint.get("name", "Unknown endpoint"),
                "request_data": filled_schema,
                "response": response
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
            for endpoint in folder_endpoints:
                if "name" in endpoint and "url" in endpoint and "schema" in endpoint:
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
            for endpoint in folder_endpoints:
                if "name" in endpoint and "url" in endpoint:
                    # Frontend endpoints have description instead of schema
                    endpoints.append(endpoint)
        
        return endpoints
    
    def _find_best_endpoint(self, prompt: str, endpoints: List[Dict]) -> Optional[Dict]:
        """
        Find the best matching endpoint based on the user prompt
        """
        try:
            if self.has_llm:
                return self._find_endpoint_with_langchain(prompt, endpoints)
        except Exception as e:
            print("Find Best Endpoint: " + traceback.format_exc())
            print(f"LLM endpoint selection failed: {str(e)}")
        # Fall back to keyword matching

        return self._find_endpoint_with_keywords(prompt, endpoints)
    
    
    def _find_endpoint_with_keywords(self, prompt: str, endpoints: List[Dict]) -> Optional[Dict]:
        """
        Fallback method using keyword matching when LLM is not available
        """
        prompt_lower = prompt.lower()
        best_match = None
        highest_score = 0
        
        for endpoint in endpoints:
            name = endpoint.get("name", "").lower()
            # Use description from both backend and frontend endpoints
            description = endpoint.get("description", "").lower()
            
            # Simple keyword matching score
            score = 0
            if name in prompt_lower:
                score += 5
            
            name_words = name.split()
            for word in name_words:
                if len(word) > 3 and word in prompt_lower:  # Only consider words longer than 3 chars
                    score += 1
            
            if description:
                description_words = description.split()
                for word in description_words:
                    if len(word) > 3 and word in prompt_lower:
                        score += 0.5
            
            if score > highest_score:
                highest_score = score
                best_match = endpoint
        
        # Only return if we have a reasonable match
        if highest_score >= 1:
            return best_match
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
        endpoint_chain = self.endpoint_selection_prompt | self.llm
        
        # Run the chain
        try:
            result = endpoint_chain.invoke({"endpoints_json": endpoints_json, "user_prompt": prompt})

            endpoint_selection = self.endpoint_parser.parse(result["endpoint_selection"])
            
            # Only proceed if a suitable endpoint was found
            if endpoint_selection.suitable:
                # Find the endpoint with matching name
                for endpoint in endpoints:
                    if endpoint.get("name", "") == endpoint_selection.endpoint_name:
                        return endpoint
        except Exception as e:
            print(f"Error in endpoint selection: {str(e)}")
            # Check if error is due to insufficient balance
            error_str = str(e)
            if "Insufficient Balance" in error_str or "402" in error_str:
                print("Detected insufficient balance, falling back to keyword matching")
                return self._find_endpoint_with_keywords(prompt, endpoints)
            return None
        
        return None
    
    
    def _fill_schema(self, prompt: str, endpoint: Dict) -> Dict:
        """
        Fill in the endpoint schema based on the user prompt
        """
        # Only applicable for backend endpoints
        if endpoint.get("endpoint_type") == "frontend":
            return {}
            
        schema = endpoint.get("schema", {})
        schema = json.loads(schema)
        if not schema:
            return {"error": "Endpoint has no schema defined", "status": 400}
        
        if self.has_llm:
            return self._fill_schema_with_langchain(prompt, endpoint)
        else:
            # Use example schema as a template if available
            example = schema.get("example", {})
            if example:
                return self._simple_fill_from_example(prompt, example)
            
            # If no example is available, try to extract parameters based on property definitions
            return self._simple_fill_from_properties(prompt, schema.get("properties", {}))
    
    
    def _fill_schema_with_langchain(self, prompt: str, endpoint: Dict) -> Dict:
        """
        Use LangChain to fill in the schema based on the user prompt
        """
        schema = endpoint.get("schema", {})
        schema = json.loads(schema)
        schema_json = json.dumps(schema, indent=2)
        
        # Get example if available - Setting as schema itself for now
        example = json.loads(endpoint.get("schema", {}))
        example_json = json.dumps(example, indent=2) if example else "No example available"
        
        # Create a chain for schema filling
        schema_chain = self.schema_filling_prompt | self.llm
        
        try:
            result = schema_chain.invoke({
                "endpoint_name": endpoint.get("name", ""),
                "schema_json": schema_json,
                "example_json": example_json,
                "user_prompt": prompt
            })
            
            # Parse the result as JSON
            try:
                filled_schema = json.loads(result["text"])
                return filled_schema
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the text
                import re
                json_match = re.search(r'```json\n(.*?)\n```', result["text"], re.DOTALL)
                if json_match:
                    try:
                        filled_schema = json.loads(json_match.group(1))
                        return filled_schema
                    except:
                        pass
                
                # Fallback to simple methods if LLM approach fails
                if example:
                    return self._simple_fill_from_example(prompt, example)
                else:
                    return self._simple_fill_from_properties(prompt, schema.get("properties", {}))
        
        except Exception as e:
            print(f"Error in schema filling: {str(e)}")
            
            # Check if error is due to insufficient balance
            error_str = str(e)
            if "Insufficient Balance" in error_str or "402" in error_str:
                print("Detected insufficient balance in schema filling, using fallback method")
            
            # Fallback to simple methods
            if example:
                return self._simple_fill_from_example(prompt, example)
            else:
                return self._simple_fill_from_properties(prompt, schema.get("properties", {}))
            
    
    def _simple_fill_from_example(self, prompt: str, example: Dict) -> Dict:
        """
        Simple parameter filling using regex and example values as fallbacks
        """
        import re
        result = {}
        
        for key, value in example.items():
            # Try to extract values based on key name
            pattern = rf"{key}[:\s]+([^,\.\s]+)"
            match = re.search(pattern, prompt, re.IGNORECASE)
            
            if match:
                extracted_value = match.group(1)
                # Type conversion based on example value type
                if isinstance(value, int):
                    try:
                        result[key] = int(extracted_value)
                    except ValueError:
                        result[key] = value  # Use example value as fallback
                elif isinstance(value, float):
                    try:
                        result[key] = float(extracted_value)
                    except ValueError:
                        result[key] = value
                else:
                    result[key] = extracted_value
            else:
                # Use example value if we can't extract from prompt
                result[key] = value
        
        return result
    
    def _simple_fill_from_properties(self, prompt: str, properties: Dict) -> Dict:
        """
        Simple parameter filling using regex and property type hints
        """
        import re
        result = {}
        
        for key, prop_info in properties.items():
            # Try to extract values based on key name
            pattern = rf"{key}[:\s]+([^,\.\s]+)"
            match = re.search(pattern, prompt, re.IGNORECASE)
            
            if match:
                extracted_value = match.group(1)
                # Type conversion based on property type
                prop_type = prop_info.get("type", "string")
                
                if prop_type == "integer":
                    try:
                        result[key] = int(extracted_value)
                    except ValueError:
                        result[key] = 0
                elif prop_type == "number":
                    try:
                        result[key] = float(extracted_value)
                    except ValueError:
                        result[key] = 0.0
                elif prop_type == "boolean":
                    result[key] = extracted_value.lower() in ["true", "yes", "1"]
                else:
                    result[key] = extracted_value
            else:
                # Use default value if defined
                if "default" in prop_info:
                    result[key] = prop_info["default"]
                else:
                    # Use type-based defaults
                    prop_type = prop_info.get("type", "string")
                    if prop_type == "string":
                        result[key] = ""
                    elif prop_type == "integer" or prop_type == "number":
                        result[key] = 0
                    elif prop_type == "boolean":
                        result[key] = False
                    elif prop_type == "array":
                        result[key] = []
                    elif prop_type == "object":
                        result[key] = {}
        
        return result
    
    def _send_request(self, endpoint: Dict, data: Dict) -> Dict:
        # Only execute for backend endpoints
        if endpoint.get("endpoint_type") == "frontend":
            return {"error": "Cannot execute frontend endpoints"}
            
        url = endpoint.get("url", "")
        print("send req to:" + url)
        method = endpoint.get("method", "POST").upper()

        print(f"Sending {method} request to {url} with data: {data}")  # Debugging

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

            print(f"Response Status: {response.status_code}")
            print(f"Response Data: {response.text}")

            if 200 <= response.status_code < 300:
                try:
                    return response.json()
                except json.JSONDecodeError:
                    return {"text": response.text}
            else:
                return {
                    "error": f"Request failed with status code: {response.status_code}",
                    "details": response.text
                }
        except Exception as e:
            print("Send Request: " + traceback.format_exc())
            return {"error": f"Request failed: {str(e)}"}


@csrf_exempt
def process_prompt(request):
    try:
        data = json.loads(request.body)
        prompt = data.get("prompt")
        api_key = data.get("api_key")
        
        if not prompt or not api_key:
            return JsonResponse(
                {"message": "Missing prompt or API key"},
                status=400
            )
        
        # Initialize the agent
        agent = EndpointAgent(
            users_collection=users_collection,
            backend_collection=backend_collection,
            frontend_collection=frontend_collection,
            deepseek_api_key=settings.DEEPSEEK_API_KEY,
        )
        
        # Process the prompt
        result = agent.process_prompt(prompt, api_key)
        
        return JsonResponse(result, status=result.get("status", 200))
    except Exception as error:
        print("process_prompt: " + traceback.format_exc())
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