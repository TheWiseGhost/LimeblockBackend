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
import uuid

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
        code = data.get("code", " ")

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
            "tokens": 250000,
            "api_key": api_key,
            "token_tracking": {},
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
                    "tokens": user.get('tokens', 0),
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

    
"""
Example request schema for add_new_folder_and_page:
{
    "user_id": "{user_id}",
    "folder_name": "Documentation",
    "page": {
        "name": "Getting Started",
        "url": "https://docs.example.com/getting-started",
        "description": "Introduction to the platform and setup instructions"
    }
}
"""
@csrf_exempt
def add_new_folder_and_page(request):
    try:
        # Parse the incoming JSON request body
        data = json.loads(request.body)
        user_id = data.get("user_id")
        folder_name = data.get("folder_name")
        page_data = data.get("page")

        # Validate required fields
        if not all([user_id, folder_name, page_data]):
            return JsonResponse(
                {"message": "Missing required fields: user_id, folder_name, and page data"},
                status=400
            )

        # Validate page data has required fields
        if not all(key in page_data for key in ["name", "url", "description"]):
            return JsonResponse(
                {"message": "Page data missing required fields: name, url, description"},
                status=400
            )

        # Convert string ID to ObjectId for MongoDB query
        try:
            user_object_id = ObjectId(user_id)
        except:
            return JsonResponse(
                {"message": "Invalid user ID format"},
                status=400
            )

        # Find the user and their frontend
        user = users_collection.find_one({"_id": user_object_id})
        if not user:
            return JsonResponse(
                {"warning": "User not found"},
                status=404
            )
            
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
            
        # Get existing frontend info
        frontend = frontend_collection.find_one({"_id": frontend_object_id})
        if not frontend:
            return JsonResponse(
                {"warning": "Frontend configuration not found"},
                status=404
            )
            
        # Generate a unique ID for the new folder
        folder_id = f"folder_{uuid.uuid4().hex}"
        
        # Generate a unique ID for the new page
        page_id = f"page_{uuid.uuid4().hex}"
        
        # Create new folder with page
        new_folder = {
            "id": folder_id,
            "name": folder_name,
            "endpoints": [{
                "id": page_id,
                "name": page_data["name"],
                "url": page_data["url"],
                "description": page_data["description"],
                "num_hits": 0
            }]
        }
        
        # Get existing folders or initialize empty array
        folders = frontend.get("folders", [])
        
        # Add the new folder
        folders.append(new_folder)
        
        # Update the frontend document
        result = frontend_collection.update_one(
            {"_id": frontend_object_id},
            {"$set": {"folders": folders}}
        )
        
        if result.modified_count > 0:
            return JsonResponse({
                "success": True, 
                "message": "Folder and page added successfully",
                "folder_id": folder_id,
                "page_id": page_id
            }, status=200)
        else:
            return JsonResponse(
                {"success": False, "message": "Failed to update frontend"},
                status=500
            )
    
    except Exception as error:
        print("Error adding folder and page:", error)
        return JsonResponse(
            {"message": "Internal Server Error", "error": str(error)},
            status=500
        )



"""
Example request schema for add_new_page:
{
    "user_id": "6f7c232d307e0b2e1c17d27",
    "folder_id": "folder_1743696478d81",
    "page": {
        "name": "API Reference",
        "url": "https://docs.example.com/api-reference",
        "description": "Complete API documentation with examples"
    }
}
"""
@csrf_exempt
def add_new_page(request):
    try:
        # Parse the incoming JSON request body
        data = json.loads(request.body)
        user_id = data.get("user_id")
        folder_id = data.get("folder_id")
        page_data = data.get("page")

        # Validate required fields
        if not all([user_id, folder_id, page_data]):
            return JsonResponse(
                {"message": "Missing required fields: user_id, folder_id, and page data"},
                status=400
            )

        # Validate page data has required fields
        if not all(key in page_data for key in ["name", "url", "description"]):
            return JsonResponse(
                {"message": "Page data missing required fields: name, url, description"},
                status=400
            )

        # Convert string ID to ObjectId for MongoDB query
        try:
            user_object_id = ObjectId(user_id)
        except:
            return JsonResponse(
                {"message": "Invalid user ID format"},
                status=400
            )

        # Find the user and their frontend
        user = users_collection.find_one({"_id": user_object_id})
        if not user:
            return JsonResponse(
                {"warning": "User not found"},
                status=404
            )
            
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
            
        # Get existing frontend info
        frontend = frontend_collection.find_one({"_id": frontend_object_id})
        if not frontend:
            return JsonResponse(
                {"warning": "Frontend configuration not found"},
                status=404
            )
            
        # Get existing folders
        folders = frontend.get("folders", [])
        
        # Find the specified folder
        folder_index = next((i for i, folder in enumerate(folders) if folder.get("id") == folder_id), None)
        
        if folder_index is None:
            return JsonResponse(
                {"warning": f"Folder with ID {folder_id} not found"},
                status=404
            )
            
        # Generate a unique ID for the new page
        page_id = f"page_{uuid.uuid4().hex}"
        
        # Create new page
        new_page = {
            "id": page_id,
            "name": page_data["name"],
            "url": page_data["url"],
            "description": page_data["description"],
            "num_hits": 0
        }
        
        # Get existing pages or initialize empty array
        pages = folders[folder_index].get("endpoints", [])
        
        # Add the new page
        pages.append(new_page)
        
        # Update the folder's pages
        folders[folder_index]["pages"] = pages
        
        # Update the frontend document
        result = frontend_collection.update_one(
            {"_id": frontend_object_id},
            {"$set": {"folders": folders}}
        )
        
        if result.modified_count > 0:
            return JsonResponse({
                "success": True, 
                "message": "Page added successfully",
                "page_id": page_id
            }, status=200)
        else:
            return JsonResponse(
                {"success": False, "message": "Failed to update frontend"},
                status=500
            )
    
    except Exception as error:
        print("Error adding page:", error)
        return JsonResponse(
            {"message": "Internal Server Error", "error": str(error)},
            status=500
        )


"""
Example request schema for edit_page:
{
    "user_id": "6f7c232d307e0b2e1c17d27",
    "folder_id": "folder_1743696478d81",
    "page_id": "page_1743694957289",
    "page_updates": {
        "name": "Updated Documentation",
        "url": "https://docs.example.com/v2/documentation",
        "description": "Revised documentation with new examples and clearer instructions"
    }
}
"""
@csrf_exempt
def edit_page(request):
    try:
        # Parse the incoming JSON request body
        data = json.loads(request.body)
        user_id = data.get("user_id")
        folder_id = data.get("folder_id")
        page_id = data.get("page_id")
        page_updates = data.get("page_updates")

        # Validate required fields
        if not all([user_id, folder_id, page_id, page_updates]):
            return JsonResponse(
                {"message": "Missing required fields: user_id, folder_id, page_id, and page_updates"},
                status=400
            )

        # Validate at least one field is being updated
        valid_update_fields = ["name", "url", "description"]
        if not any(field in page_updates for field in valid_update_fields):
            return JsonResponse(
                {"message": "No valid fields to update"},
                status=400
            )

        # Convert string ID to ObjectId for MongoDB query
        try:
            user_object_id = ObjectId(user_id)
        except:
            return JsonResponse(
                {"message": "Invalid user ID format"},
                status=400
            )

        # Find the user and their frontend
        user = users_collection.find_one({"_id": user_object_id})
        if not user:
            return JsonResponse(
                {"warning": "User not found"},
                status=404
            )
            
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
            
        # Get existing frontend info
        frontend = frontend_collection.find_one({"_id": frontend_object_id})
        if not frontend:
            return JsonResponse(
                {"warning": "Frontend configuration not found"},
                status=404
            )
            
        # Get existing folders
        folders = frontend.get("folders", [])
        
        # Find the specified folder
        folder_index = next((i for i, folder in enumerate(folders) if folder.get("id") == folder_id), None)
        
        if folder_index is None:
            return JsonResponse(
                {"warning": f"Folder with ID {folder_id} not found"},
                status=404
            )
            
        # Find the specified page
        pages = folders[folder_index].get("endpoints", [])
        page_index = next((i for i, page in enumerate(pages) if page.get("id") == page_id), None)
        
        if page_index is None:
            return JsonResponse(
                {"warning": f"Page with ID {page_id} not found in folder {folder_id}"},
                status=404
            )
            
        # Update only the provided fields
        for field in valid_update_fields:
            if field in page_updates:
                pages[page_index][field] = page_updates[field]
        
        # Update the folder's pages
        folders[folder_index]["pages"] = pages
        
        # Update the frontend document
        result = frontend_collection.update_one(
            {"_id": frontend_object_id},
            {"$set": {"folders": folders}}
        )
        
        if result.modified_count > 0:
            return JsonResponse({
                "success": True, 
                "message": "Page updated successfully"
            }, status=200)
        else:
            return JsonResponse(
                {"success": True, "message": "No changes made to page"},
                status=200
            )
    
    except Exception as error:
        print("Error editing page:", error)
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
    


"""
Example request schema for add_new_folder_and_endpoint:
{
    "user_id": "{user_id}",
    "folder_name": "Authentication",
    "endpoint": {
        "name": "Login",
        "url": "https://api.example.com/login",
        "method": "POST",
        "schema": {
            "username": "string",
            "password": "string"
        },
        "description": "Authenticates a user and returns a token",
        "examplePrompts": [],
        "requiredContextParams": ["user_id"],
        "instructions": "Send credentials to authenticate the user",
        "num_hits": 0
    }
}
"""
@csrf_exempt
def add_new_folder_and_endpoint(request):
    try:
        # Parse the incoming JSON request body
        data = json.loads(request.body)
        user_id = data.get("user_id")
        folder_name = data.get("folder_name")
        endpoint_data = data.get("endpoint")

        # Validate required fields
        if not all([user_id, folder_name, endpoint_data]):
            return JsonResponse(
                {"message": "Missing required fields: user_id, folder_name, and endpoint data"},
                status=400
            )

        # Validate endpoint data has required fields
        if not all(key in endpoint_data for key in ["name", "url", "method", "schema"]):
            return JsonResponse(
                {"message": "Endpoint data missing required fields: name, url, method, schema"},
                status=400
            )

        # Convert string ID to ObjectId for MongoDB query
        try:
            user_object_id = ObjectId(user_id)
        except:
            return JsonResponse(
                {"message": "Invalid user ID format"},
                status=400
            )

        # Find the user and their backend
        user = users_collection.find_one({"_id": user_object_id})
        if not user:
            return JsonResponse(
                {"warning": "User not found"},
                status=404
            )
            
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
            
        # Get existing backend info
        backend = backend_collection.find_one({"_id": backend_object_id})
        if not backend:
            return JsonResponse(
                {"warning": "Backend configuration not found"},
                status=404
            )
            
        # Generate a unique ID for the new folder
        folder_id = f"folder_{uuid.uuid4().hex}"
        
        # Generate a unique ID for the new endpoint
        endpoint_id = f"endpoint_{uuid.uuid4().hex}"
        
        # Create new folder with endpoint
        new_folder = {
            "id": folder_id,
            "name": folder_name,
            "endpoints": [{
                "id": endpoint_id,
                "name": endpoint_data["name"],
                "url": endpoint_data["url"],
                "method": endpoint_data["method"],
                "schema": endpoint_data["schema"],
                "description": endpoint_data.get("description", ""),
                "params": endpoint_data.get("params", []),
                "instructions": endpoint_data.get("instructions", ""),
                "required_params": endpoint_data.get("required_params", []),
                "num_bits": endpoint_data.get("num_bits", 1)
            }]
        }
        
        # Get existing folders or initialize empty array
        folders = backend.get("folders", [])
        
        # Add the new folder
        folders.append(new_folder)
        
        # Update the backend document
        result = backend_collection.update_one(
            {"_id": backend_object_id},
            {"$set": {"folders": folders}}
        )
        
        if result.modified_count > 0:
            return JsonResponse({
                "success": True, 
                "message": "Folder and endpoint added successfully",
                "folder_id": folder_id,
                "endpoint_id": endpoint_id
            }, status=200)
        else:
            return JsonResponse(
                {"success": False, "message": "Failed to update backend"},
                status=500
            )
    
    except Exception as error:
        print("Error adding folder and endpoint:", error)
        return JsonResponse(
            {"message": "Internal Server Error", "error": str(error)},
            status=500
        )


"""
Example request schema for add_new_endpoint:
{
    "user_id": "6f7c232d307e0b2e1c17d27",
    "folder_id": "folder_1743696478d81",
    "endpoint": {
        "name": "Login",
        "url": "https://api.example.com/login",
        "method": "POST",
        "schema": {
            "username": "string",
            "password": "string"
        },
        "description": "Authenticates a user and returns a token",
        "examplePrompts": [],
        "requiredContextParams": ["user_id"],
        "instructions": "Send credentials to authenticate the user",
        "num_hits": 0
    }
}
"""
@csrf_exempt
def add_new_endpoint(request):
    try:
        # Parse the incoming JSON request body
        data = json.loads(request.body)
        user_id = data.get("user_id")
        folder_id = data.get("folder_id")
        endpoint_data = data.get("endpoint")

        # Validate required fields
        if not all([user_id, folder_id, endpoint_data]):
            return JsonResponse(
                {"message": "Missing required fields: user_id, folder_id, and endpoint data"},
                status=400
            )

        # Validate endpoint data has required fields
        if not all(key in endpoint_data for key in ["name", "url", "method", "schema"]):
            return JsonResponse(
                {"message": "Endpoint data missing required fields: name, url, method, schema"},
                status=400
            )

        # Convert string ID to ObjectId for MongoDB query
        try:
            user_object_id = ObjectId(user_id)
        except:
            return JsonResponse(
                {"message": "Invalid user ID format"},
                status=400
            )

        # Find the user and their backend
        user = users_collection.find_one({"_id": user_object_id})
        if not user:
            return JsonResponse(
                {"warning": "User not found"},
                status=404
            )
            
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
            
        # Get existing backend info
        backend = backend_collection.find_one({"_id": backend_object_id})
        if not backend:
            return JsonResponse(
                {"warning": "Backend configuration not found"},
                status=404
            )
            
        # Get existing folders
        folders = backend.get("folders", [])
        
        # Find the specified folder
        folder_index = next((i for i, folder in enumerate(folders) if folder.get("id") == folder_id), None)
        
        if folder_index is None:
            return JsonResponse(
                {"warning": f"Folder with ID {folder_id} not found"},
                status=404
            )
            
        # Generate a unique ID for the new endpoint
        endpoint_id = f"endpoint_{uuid.uuid4().hex}"
        
        # Create new endpoint
        new_endpoint = {
            "id": endpoint_id,
            "name": endpoint_data["name"],
            "url": endpoint_data["url"],
            "method": endpoint_data["method"],
            "schema": endpoint_data["schema"],
            "description": endpoint_data.get("description", ""),
            "params": endpoint_data.get("params", []),
            "instructions": endpoint_data.get("instructions", ""),
            "required_params": endpoint_data.get("required_params", []),
            "num_bits": endpoint_data.get("num_bits", 1)
        }
        
        # Get existing endpoints or initialize empty array
        endpoints = folders[folder_index].get("endpoints", [])
        
        # Add the new endpoint
        endpoints.append(new_endpoint)
        
        # Update the folder's endpoints
        folders[folder_index]["endpoints"] = endpoints
        
        # Update the backend document
        result = backend_collection.update_one(
            {"_id": backend_object_id},
            {"$set": {"folders": folders}}
        )
        
        if result.modified_count > 0:
            return JsonResponse({
                "success": True, 
                "message": "Endpoint added successfully",
                "endpoint_id": endpoint_id
            }, status=200)
        else:
            return JsonResponse(
                {"success": False, "message": "Failed to update backend"},
                status=500
            )
    
    except Exception as error:
        print("Error adding endpoint:", error)
        return JsonResponse(
            {"message": "Internal Server Error", "error": str(error)},
            status=500
        )



"""
Example request schema for edit_endpoint:
{
    "user_id": "6f7c232d307e0b2e1c17d27",
    "folder_id": "folder_1743696478d81",
    "endpoint_id": "endpoint_1743694957289",
    "endpoint_updates": {
        "name": "Login",
        "url": "https://api.example.com/login",
        "method": "POST",
        "schema": {
            "username": "string",
            "password": "string"
        },
        "description": "Authenticates a user and returns a token",
        "examplePrompts": [],
        "requiredContextParams": ["user_id"],
        "instructions": "Send credentials to authenticate the user",
        "num_hits": 0
    }
}
"""


@csrf_exempt
def edit_endpoint(request):
    try:
        # Parse the incoming JSON request body
        data = json.loads(request.body)
        user_id = data.get("user_id")
        folder_id = data.get("folder_id")
        endpoint_id = data.get("endpoint_id")
        endpoint_updates = data.get("endpoint_updates")

        # Validate required fields
        if not all([user_id, folder_id, endpoint_id, endpoint_updates]):
            return JsonResponse(
                {"message": "Missing required fields: user_id, folder_id, endpoint_id, and endpoint_updates"},
                status=400
            )

        # Validate at least one field is being updated
        valid_update_fields = ["name", "url", "method", "schema", "description", 
                              "params", "instructions", "required_params", "num_bits"]
        if not any(field in endpoint_updates for field in valid_update_fields):
            return JsonResponse(
                {"message": "No valid fields to update"},
                status=400
            )

        # Convert string ID to ObjectId for MongoDB query
        try:
            user_object_id = ObjectId(user_id)
        except:
            return JsonResponse(
                {"message": "Invalid user ID format"},
                status=400
            )

        # Find the user and their backend
        user = users_collection.find_one({"_id": user_object_id})
        if not user:
            return JsonResponse(
                {"warning": "User not found"},
                status=404
            )
            
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
            
        # Get existing backend info
        backend = backend_collection.find_one({"_id": backend_object_id})
        if not backend:
            return JsonResponse(
                {"warning": "Backend configuration not found"},
                status=404
            )
            
        # Get existing folders
        folders = backend.get("folders", [])
        
        # Find the specified folder
        folder_index = next((i for i, folder in enumerate(folders) if folder.get("id") == folder_id), None)
        
        if folder_index is None:
            return JsonResponse(
                {"warning": f"Folder with ID {folder_id} not found"},
                status=404
            )
            
        # Find the specified endpoint
        endpoints = folders[folder_index].get("endpoints", [])
        endpoint_index = next((i for i, endpoint in enumerate(endpoints) if endpoint.get("id") == endpoint_id), None)
        
        if endpoint_index is None:
            return JsonResponse(
                {"warning": f"Endpoint with ID {endpoint_id} not found in folder {folder_id}"},
                status=404
            )
            
        # Update only the provided fields
        for field in valid_update_fields:
            if field in endpoint_updates:
                endpoints[endpoint_index][field] = endpoint_updates[field]
        
        # Update the folder's endpoints
        folders[folder_index]["endpoints"] = endpoints
        
        # Update the backend document
        result = backend_collection.update_one(
            {"_id": backend_object_id},
            {"$set": {"folders": folders}}
        )
        
        if result.modified_count > 0:
            return JsonResponse({
                "success": True, 
                "message": "Endpoint updated successfully"
            }, status=200)
        else:
            return JsonResponse(
                {"success": True, "message": "No changes made to endpoint"},
                status=200
            )
    
    except Exception as error:
        print("Error editing endpoint:", error)
        return JsonResponse(
            {"message": "Internal Server Error", "error": str(error)},
            status=500
        )


"""
Example request schema for delete_endpoint:
{
    "user_id": "6f7c232d307e0b2e1c17d27",
    "folder_id": "folder_1743696478d81",
    "endpoint_id": "endpoint_1743694957289"
}
"""
@csrf_exempt
def delete_endpoint(request):
    try:
        # Parse the incoming JSON request body
        data = json.loads(request.body)
        user_id = data.get("user_id")
        folder_id = data.get("folder_id")
        endpoint_id = data.get("endpoint_id")

        # Validate required fields
        if not all([user_id, folder_id, endpoint_id]):
            return JsonResponse(
                {"message": "Missing required fields: user_id, folder_id, and endpoint_id"},
                status=400
            )

        # Convert string ID to ObjectId for MongoDB query
        try:
            user_object_id = ObjectId(user_id)
        except:
            return JsonResponse(
                {"message": "Invalid user ID format"},
                status=400
            )

        # Find the user and their backend
        user = users_collection.find_one({"_id": user_object_id})
        if not user:
            return JsonResponse(
                {"warning": "User not found"},
                status=404
            )
            
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
            
        # Get existing backend info
        backend = backend_collection.find_one({"_id": backend_object_id})
        if not backend:
            return JsonResponse(
                {"warning": "Backend configuration not found"},
                status=404
            )
            
        # Get existing folders
        folders = backend.get("folders", [])
        
        # Find the specified folder
        folder_index = next((i for i, folder in enumerate(folders) if folder.get("id") == folder_id), None)
        
        if folder_index is None:
            return JsonResponse(
                {"warning": f"Folder with ID {folder_id} not found"},
                status=404
            )
            
        # Find and remove the specified endpoint
        endpoints = folders[folder_index].get("endpoints", [])
        filtered_endpoints = [endpoint for endpoint in endpoints if endpoint.get("id") != endpoint_id]
        
        if len(filtered_endpoints) == len(endpoints):
            return JsonResponse(
                {"warning": f"Endpoint with ID {endpoint_id} not found in folder {folder_id}"},
                status=404
            )
            
        # Update the folder's endpoints
        folders[folder_index]["endpoints"] = filtered_endpoints
        
        # Update the backend document
        result = backend_collection.update_one(
            {"_id": backend_object_id},
            {"$set": {"folders": folders}}
        )
        
        if result.modified_count > 0:
            return JsonResponse({
                "success": True, 
                "message": "Endpoint deleted successfully"
            }, status=200)
        else:
            return JsonResponse(
                {"success": False, "message": "Failed to delete endpoint"},
                status=500
            )
    
    except Exception as error:
        print("Error deleting endpoint:", error)
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
                
                If an endpoint matches my request, tell me that's the endpoint I should hit. 
                
                {format_instructions}
                
                If none of the endpoints can fulfill the request, set suitable to false.
                """,
                input_variables=["endpoints_json", "user_prompt", "user_context"],
                partial_variables={"format_instructions": self.endpoint_parser.get_format_instructions()}
            )
            
            self.schema_filling_prompt = PromptTemplate(
                template="""
                Example Schema:
                {schema_json}  
               
                User Prompt: {user_prompt}
                Here is some context: {user_context}

                Return a valid JSON object and nothing else. This schema is going to be sent to an endpoint called {endpoint_name} and it's described as {endpoint_description}.
                The instructions are {endpoint_instructions}

                If prompt doesn't apply to the schema properly or doesn't fit the endpoint or isn't specific on what to do, please just message me back "No action", if not
                do the steps below:

                Fill in values surrounded by curly braces in schema like "user_id": "{{user_id}}" with things in the
                context I gave you. 

                Don't modify the structure of the schema, just change the values so it matches what I'm trying to do. 
                """,
                input_variables=["endpoint_name", "endpoint_description", "endpoint_instructions", "schema_json", "user_prompt", "user_context"]
            )

            self.response_creation_prompt = PromptTemplate(
                template="""
                Prompt: {user_prompt}
                Request processed using endpoint: {endpoint_name} that is described as {endpoint_description}

                Response from server: {server_response}

                Give a message that sums this up and make it short and to the point and 
                have nothing about the endpoint used or its name or what was done to process the user's request. just give the useful info the user needs.
                """,
                input_variables=["endpoint_name", "endpoint_description", "server_response", "user_prompt"]
            )

            self.no_endpoint_response_creation_prompt = PromptTemplate(
                template="""
                Prompt: {user_prompt}

                Give a message that responds to this as a member of company {company_name}. If you don't know what to respond give a message that 
                tells the user to check the docs or contact support or etc depending on the prompt. Make the message short and to the point. Don't make anything up.
                """,
                input_variables=["company_name", "user_prompt"]
            )
    
    def process_prompt(self, prompt: str, api_key: str, context: Optional[Dict] = None, option: Optional[str] = "Find Page"):
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
        all_endpoints = []
        if option == "Find Page":
            frontend_endpoints = self._get_all_frontend_endpoints(frontend_config) if frontend_config else []
            for endpoint in frontend_endpoints:
                endpoint["endpoint_type"] = "frontend"
                all_endpoints.append(endpoint)  
        else:   
            backend_endpoints = self._get_all_endpoints(backend_config, context) if backend_config else []
            for endpoint in backend_endpoints:
                endpoint["endpoint_type"] = "backend"
                all_endpoints.append(endpoint)

        logging.debug(f"Total available endpoints: {json.dumps(all_endpoints, indent=2)}")

        # Step 5: Find the best matching endpoint
        # print("PROMPT:" + prompt)
        # print("CONTEXT:" + str(context))
        # print("ALL_ENDPOINTS" + str(all_endpoints))
        best_endpoint = self._find_best_endpoint(prompt, context, all_endpoints)

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

            # # Step 8: Send request
            # response = self._send_request(best_endpoint, filled_schema)
            # logging.debug(f"Response from backend: {response}")

            # backend_collection.update_one(
            #     {"_id": ObjectId(backend_id), "folders.id": best_endpoint.get("folder_id", ""), "folders.endpoints.id": best_endpoint.get("id", "")},
            #     {"$inc": {"folders.$[folder].endpoints.$[endpoint].num_hits": 1}},
            #     array_filters=[
            #         {"folder.id": best_endpoint.get("folder_id", "")},
            #         {"endpoint.id": best_endpoint.get("id", "")}
            #     ]
            #  )  

            # formatted_response = self.llm.invoke(self.response_creation_prompt.format(
            #     endpoint_name=best_endpoint.get("name", ""),
            #     endpoint_description=best_endpoint.get("description", ""),
            #     user_prompt=prompt,
            #     server_response=response
            # ))

            return {
                "status": 200,
                "endpoint_type": "backend",
                "endpoint": best_endpoint,
                "schema": filled_schema,
                "prompt": prompt,
                "formatted_response": f"Please confirm this action to endpoint - {best_endpoint.get('name', '')}.\nIf you don't think this endpoint makes sense, the AI might have made a mistake. Actions may be irreversible.",
                "confirm_needed": True,
            }
        
    
    def _get_all_endpoints(self, backend_config: Dict, context: Dict = None) -> List[Dict]:
        """
        Extract all endpoints from the backend configuration and check if context satisfies required parameters

        Args:
            backend_config: The backend configuration
            context: The context dictionary containing parameters for endpoint calls
            
        Returns:
            A list of endpoints with an additional field indicating if all required context parameters are present
        """
        if not backend_config:
            return []
            
        if context is None:
            context = {}
            
        endpoints = []
        folders = backend_config.get("folders", [])

        # Process each folder
        for folder in folders:
            folder_endpoints = folder.get("endpoints", [])
            folder_id = folder.get("id", "")
            for endpoint in folder_endpoints:
                if "name" in endpoint and "url" in endpoint and "schema" in endpoint:
                    # Add folder ID to the endpoint
                    endpoint['folder_id'] = folder_id
                    
                    # Check if all required context parameters are present
                    required_params = endpoint.get("requiredContextParams", [])
                    has_all_params = True
                    
                    for param in required_params:
                        if param not in context:
                            has_all_params = False
                    
                    if has_all_params:
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
    
    def _find_best_endpoint(self, prompt: str, context: Dict, endpoints: List[Dict]) -> Optional[Dict]:
        """
        Find the best matching endpoint based on the user prompt
        """
        try:
            if self.has_llm:
                result = self._find_endpoint_with_langchain(prompt, context, endpoints)
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
    

    def _find_endpoint_with_langchain(self, prompt: str, context: Dict, endpoints: List[Dict]) -> Optional[Dict]:
        """
        Use LangChain and DeepSeek to find the best endpoint based on understanding the prompt
        """
        # Format endpoints for the LLM, including endpoint type
        endpoints_json = json.dumps([{
            "name": ep.get("name", ""),
            "description": ep.get("description", ""),
            "example_prompts": ep.get("examplePrompts", ""),
            "type": ep.get("endpoint_type", "backend")
        } for ep in endpoints], indent=2)
        
        # Create a chain for endpoint selection
        try:
            result = self.llm.invoke(self.endpoint_selection_prompt.format(
                endpoints_json=endpoints_json, 
                user_prompt=prompt,
                user_context=context
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
                endpoint_instructions=endpoint.get("instructions", ""),
                schema_json=schema_json,
                user_prompt=prompt,
                user_context=context_json
            ))

            # Extract JSON from response
            try:
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', result.content, re.DOTALL)
                filled_schema = json.loads(json_match.group(1)) if json_match else json.loads(result.content)
                
                # Ensure any remaining placeholders are replaced
                filled_schema = self.replace_context_placeholders(filled_schema, context)
                return filled_schema
            except json.JSONDecodeError:
                logging.warning("Failed to parse LLM response as JSON -> AI can't do this")
                return "AI can't do this"

        except Exception as e:
            logging.error(f"Error in schema filling: {str(e)}")
            return "AI can't do this"
            
    
    def _simple_fill_from_example(self, prompt: str, schema_dict: Dict) -> Dict:
        """
        Use the schema as-is since it's already an example
        """
        return "AI can't do this"  # Placeholder for when LLM is not available
    
    def _send_request(self, endpoint: Dict, data: Dict) -> Dict:
        # # Only execute for backend endpoints
        # if endpoint.get("endpoint_type") == "frontend":
        #     return {"error": "Cannot execute frontend endpoints"}
            
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
        option = data.get("option", "Find Page")
        
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
        result = agent.process_prompt(prompt, api_key, context, option)
        
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
    


@csrf_exempt
def commit_backend_action(request):
    try:
        data = json.loads(request.body)
        best_endpoint = data.get("endpoint")
        schema = data.get("schema")
        prompt = data.get("prompt")
        api_key = data.get("api_key")
        
        user = users_collection.find_one({"api_key": api_key})
        if not user:
            logging.error("User not found.")
            return {"error": "User not found", "status": 404}

        logging.debug(f"User found: {user}")

        # Step 2: Get backend and frontend document IDs
        backend_id = user.get("backend")

        # Initialize the agent
        agent = EndpointAgent(
            users_collection=users_collection,
            backend_collection=backend_collection,
            frontend_collection=frontend_collection,
            deepseek_api_key=f'{settings.DEEPSEEK_API_KEY}',
        )

        response = agent._send_request(best_endpoint, schema)
        logging.debug(f"Response from backend: {response}")

        backend_collection.update_one(
            {"_id": ObjectId(backend_id), "folders.id": best_endpoint.get("folder_id", ""), "folders.endpoints.id": best_endpoint.get("id", "")},
            {"$inc": {"folders.$[folder].endpoints.$[endpoint].num_hits": 1}},
            array_filters=[
                {"folder.id": best_endpoint.get("folder_id", "")},
                {"endpoint.id": best_endpoint.get("id", "")}
            ]
        )  

        formatted_response = agent.llm.invoke(agent.response_creation_prompt.format(
            endpoint_name=best_endpoint.get("name", ""),
            endpoint_description=best_endpoint.get("description", ""),
            user_prompt=prompt,
            server_response=response
        ))

        result = {
            "status": 200,
            "endpoint_type": "completed_backend",
            "endpoint_used": best_endpoint.get("name", "Unknown endpoint"),
            "request_data": schema,
            "response": response, 
            "formatted_response": str(formatted_response.content),
        }
        
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
    



@csrf_exempt
def ai_action(request):
    try:
        data = json.loads(request.body)
        context = data.get("context", {})
        endpoint_id = data.get("endpoint_id")
        folder_id = data.get("folder_id")
        prompt = data.get("prompt")
        api_key = data.get("api_key")
        formatting_needed = data.get("formatting_needed", False)
        
        if not are_tokens_remaining(api_key):
            return JsonResponse(
                {"response": "This company has reached the monthly token limit. Upgrade needed to continue using Limeblock."},
                status=403
            )
        
        user = users_collection.find_one({"api_key": api_key})
        if not user:
            logging.error("User not found.")
            return {"error": "User not found", "status": 404}

        logging.debug(f"User found: {user}")

        # Step 2: Get backend and frontend document IDs
        backend_id = user.get("backend")

        # Find the specific endpoint from the backend collection
        backend_doc = backend_collection.find_one({"_id": ObjectId(backend_id)})
        if not backend_doc:
            logging.error("Backend document not found.")
            return JsonResponse({"error": "Backend not found", "status": 404}, status=404)

        endpoint = None
        for folder in backend_doc.get("folders", []):
            if folder.get("id") == folder_id:
                for e in folder.get("endpoints", []):
                    if e.get("id") == endpoint_id:
                        endpoint = e
                        break
                break

        if not endpoint:
            logging.error(f"Endpoint not found with folder_id: {folder_id}, endpoint_id: {endpoint_id}")
            return JsonResponse({"error": "Endpoint not found", "status": 404}, status=404)

        # Initialize the agent
        agent = EndpointAgent(
            users_collection=users_collection,
            backend_collection=backend_collection,
            frontend_collection=frontend_collection,
            deepseek_api_key=f'{settings.DEEPSEEK_API_KEY}',
        )

        schema = agent._fill_schema(prompt, endpoint, context)
        print("FILLED_SCHEMA: " + str(schema))
        prompt_tokens = count_tokens(prompt)
        context_tokens = count_tokens(json.dumps(context))
        endpoint_tokens = count_tokens(json.dumps(endpoint))  # Add endpoint tokens
        input_tokens = prompt_tokens + context_tokens + endpoint_tokens
        
        # Count output tokens from the schema generation
        output_tokens = count_tokens(str(schema))

        # 88 is just the tokens we use in the prompt
        total_tokens = input_tokens + output_tokens * 4 + 88

        if isinstance(schema, str):
            print("TOTAL TOKENS USED: " + str(total_tokens))
            update_token_count(api_key, total_tokens)
            return JsonResponse({"response": schema, "status": 200})

        response = agent._send_request(endpoint, schema)
        logging.debug(f"Response from backend: {response}")
        
        # If formatting is needed, count those tokens too
        if formatting_needed:
            # Include endpoint name and description in formatting token count
            endpoint_name = endpoint.get("name", "")
            endpoint_description = endpoint.get("description", "")
            
            formatted_response = agent.llm.invoke(agent.response_creation_prompt.format(
                endpoint_name=endpoint_name,
                endpoint_description=endpoint_description,
                user_prompt=prompt,
                server_response=response
            ))
            formatted_content = str(formatted_response.content)
            
            # Count formatting input tokens (endpoint name + description + prompt + response)
            formatting_input_tokens = (count_tokens(endpoint_name) + 
                                     count_tokens(endpoint_description) + 
                                     count_tokens(prompt) + 
                                     count_tokens(str(response)))
            # Count formatting output tokens
            formatting_output_tokens = count_tokens(formatted_content)
            # 60 tokens used in prompt
            formatting_tokens = formatting_input_tokens + formatting_output_tokens * 4 + 50
            total_tokens += formatting_tokens
        else:
            formatted_content = None
            formatting_tokens = 0


        backend_collection.update_one(
            {"_id": ObjectId(backend_id), "folders.id": folder_id, "folders.endpoints.id": endpoint_id},
            {"$inc": {"folders.$[folder].endpoints.$[endpoint].num_hits": 1}},
            array_filters=[
                {"folder.id": folder_id},
                {"endpoint.id": endpoint_id}
            ]
        )  
        
        print("TOTAL TOKENS USED: " + str(total_tokens))
        # Update token count with total tokens used
        update_token_count(api_key, total_tokens)
        
        if formatting_needed:
            result = {
                "status": 200,
                "endpoint_type": "completed_backend",
                "endpoint_used": endpoint.get("name", "Unknown endpoint"),
                "request_data": schema,
                "response": response, 
                "formatted_response": formatted_content,
            }
        else:
            result = {
                "status": 200,
                "endpoint_type": "completed_backend",
                "endpoint_used": endpoint.get("name", "Unknown endpoint"),
                "request_data": schema,
                "response": response,
            }
        
        # Ensure we return 200 status even if no endpoint is found
        if "status" in result and result["status"] != 200:
            result_status = result.pop("status", 200)
            return JsonResponse(result, status=result_status)
        else:
            return JsonResponse(result, status=200)
    except Exception as error:
        logging.error("ai_action error: " + traceback.format_exc())
        return JsonResponse(
            {"message": "Internal Server Error", "error": str(error)},
            status=500
        )
    


import tiktoken  # For token counting
from deepseek_tokenizer import ds_token

def count_tokens(text):
    try:
        # encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        # return len(encoding.encode(text))
        return len(ds_token.encode(str(text)))
    except:
        print("Error counting tokens with tiktoken, falling back to approximate method.")
        # Fallback: approximate token count (1 token ≈ 4 characters)
        return max(1, len(text) // 4)

def update_token_count(api_key, token_count):
    """
    Updates the token balance by deducting the used tokens.
    Ensures balance never goes below 0.
    
    Args:
        api_key: API key
        token_count: Number of tokens to deduct
    
    Returns:
        dict: {'success': bool, 'new_balance': int, 'tokens_deducted': int}
    """
    try:
        # Find the user document by API key
        user = users_collection.find_one({"api_key": api_key})
        if not user:
            logging.warning(f"No user found with API key: {api_key}")
            return {'success': False, 'new_balance': 0, 'tokens_deducted': 0}
        
        # Get current token balance (default to 0 if not set)
        current_balance = user.get("tokens", 0)
        
        # Calculate how many tokens to actually deduct
        # If they have 10 tokens and try to use 207, deduct all 10
        tokens_to_deduct = min(current_balance, token_count)
        new_balance = max(0, current_balance - token_count)
        
        # Update the token balance
        users_collection.update_one(
            {"api_key": api_key},
            {"$set": {"tokens": new_balance}}
        )
        
        # Still track usage for analytics (optional - keep monthly tracking for stats)
        current_month = datetime.datetime.now().strftime("%Y-%m")
        users_collection.update_one(
            {"api_key": api_key},
            {"$inc": {f"token_tracking.{current_month}": token_count}}
        )
        
        return {
            'success': True, 
            'new_balance': new_balance, 
            'tokens_deducted': tokens_to_deduct,
            'tokens_requested': token_count
        }
        
    except Exception as e:
        logging.error(f"Error updating token balance: {traceback.format_exc()}")
        return {'success': False, 'new_balance': 0, 'tokens_deducted': 0}


def add_tokens_to_balance(api_key, tokens_to_add):
    """
    Adds tokens to a user's balance (for purchases).
    
    Args:
        api_key: API key
        tokens_to_add: Number of tokens to add to balance
    
    Returns:
        dict: {'success': bool, 'new_balance': int}
    """
    try:
        user = users_collection.find_one({"api_key": api_key})
        if not user:
            logging.warning(f"No user found with API key: {api_key}")
            return {'success': False, 'new_balance': 0}
        
        current_balance = user.get("tokens", 0)
        new_balance = current_balance + tokens_to_add
        
        users_collection.update_one(
            {"api_key": api_key},
            {"$set": {"tokens": new_balance}}
        )
        
        return {'success': True, 'new_balance': new_balance}
        
    except Exception as e:
        logging.error(f"Error adding tokens to balance: {traceback.format_exc()}")
        return {'success': False, 'new_balance': 0}


@csrf_exempt
def get_token_stats(request):
    """
    Get token statistics for a specific user over a number of months.
    
    Args:
        users_collection: MongoDB collection containing user data
        user_id: User ID
        months: Number of months to include in the statistics (default 6)
    
    Returns:
        dict: Monthly token statistics
    """
    try:
        data = json.loads(request.body)
        user_id = data.get("user_id")
        months = data.get("months", 6)

        user = users_collection.find_one({"_id": ObjectId(user_id)})
        if not user or "token_tracking" not in user:
            return JsonResponse({"error": "No token data found"}, status=404)
        
        current_balance = user.get("tokens", 0)
        
        # Get the current month and previous months
        current_date = datetime.datetime.now()
        month_stats = {}

        for i in range(months):
            month_date = current_date.replace(day=1) - relativedelta(months=i)
            month_key = month_date.strftime("%Y-%m")
            
            # Get token count for this month
            month_stats[month_key] = user.get("token_tracking", {}).get(month_key, 0)

        return JsonResponse(
            {"success": True, "remaining": current_balance, "usage_stats": month_stats},
            status=200
        )

    except Exception as error:
        return JsonResponse(
            {"message": "Internal Server Error", "error": str(error)},
            status=500
        )

def are_tokens_remaining(api_key, estimated_token_usage=0):
    """
    Check if user has sufficient tokens in their balance.
    
    Args:
        api_key: User's API key
        estimated_token_usage: Estimated tokens for the current request (optional)
    
    Returns:
        dict: {'has_tokens': bool, 'current_balance': int, 'estimated_usage': int}
    """
    try:
        if not api_key:
            return {'has_tokens': False, 'current_balance': 0, 'estimated_usage': estimated_token_usage}
        
        user = users_collection.find_one({"api_key": api_key})
        if not user:
            logging.error("User not found in token check.")
            return {'has_tokens': False, 'current_balance': 0, 'estimated_usage': estimated_token_usage}
        
        current_balance = user.get("tokens", 0)
        
        # Check if user has any tokens (they can use more than they have, balance just goes to 0)
        has_tokens = current_balance > 0
        
        return {
            'has_tokens': has_tokens, 
            'current_balance': current_balance, 
            'estimated_usage': estimated_token_usage
        }
        
    except Exception as error:
        logging.error("Token check error: " + traceback.format_exc())
        return {'has_tokens': False, 'current_balance': 0, 'estimated_usage': estimated_token_usage}

def get_user_token_balance(api_key):
    """
    Get the current token balance for a user.
    
    Args:
        api_key: User's API key
    
    Returns:
        int: Current token balance (0 if user not found)
    """
    try:
        if not api_key:
            return 0
            
        user = users_collection.find_one({"api_key": api_key})
        if not user:
            return 0
            
        return user.get("tokens", 0)
        
    except Exception as error:
        logging.error(f"Error getting user token balance: {traceback.format_exc()}")
        return 0


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



@csrf_exempt
def are_maus_remaining(request):
    try:
        data = json.loads(request.body)
        api_key = data.get("api_key")

        
        if not api_key:
            return JsonResponse(
                {"message": "Missing API key"},
                status=400
            )
        
        user = users_collection.find_one({"api_key": api_key})
        if not user:
            logging.error("User not found.")
            return {"error": "User not found", "status": 404}

        
        current_date = datetime.datetime.now()

        month_date = current_date.replace(day=1) - relativedelta(months=0)
        month_key = month_date.strftime("%Y-%m")
        
        # Get MAU count for this month
        curr_month_mau_count = user.get("mau_count", {}).get(month_key, 0)
        
        # Define the time threshold (1.5 months ago)
        time_threshold = datetime.datetime.today() - timedelta(days=45)

        last_paid = user.get("last_paid")

        if last_paid and last_paid < time_threshold:
            # Update plan to 'free' in MongoDB
            users_collection.update_one(
                {"api_key": api_key},
                {"$set": {"plan": "free"}}
            )
            user['plan'] = "free"  # Reflect this in the response

        if user['plan'] == "business":
            user_maus = 1000
        elif user['plan'] == "startup":
            user_maus = 100
        elif user['plan'] == "enterprise":
            user_maus = 5000
        else:
            user_maus = 20
        
        if (user_maus < curr_month_mau_count):
            return JsonResponse({'valid': False}, status=200)
        else:
            return JsonResponse({'valid': True}, status=200)
        
    except Exception as error:
        logging.error("process_prompt error: " + traceback.format_exc())
        return JsonResponse(
            {"message": "Internal Server Error", "error": str(error)},
            status=500
        )


@csrf_exempt
def test_endpoint(request):
    """
    Django view function to test if an endpoint is accessible
    
    Expected request JSON:
    {
        "endpoint_name": "name_of_endpoint",
        "api_key": "user_api_key"
    }
    
    Returns:
        JsonResponse with test results
    """
    try:
        data = json.loads(request.body)
        target_endpoint = data.get("endpoint")
        
        if not target_endpoint:
            return JsonResponse({
                "success": False,
                "message": f"Endpoint not recieved to test",
                "error": f"Endpoint not recieved to test",
                "status": 404
            }, status=404)
        
        # Try to make a basic request to test accessibility
        url = target_endpoint.get("url", "")
        method = target_endpoint.get("method", "GET").upper()
        
        try:
            import requests
            
            # Send a minimal request - just testing connectivity, not functionality
            if method == "GET":
                response = requests.get(url, timeout=5)
            else:
                # For non-GET methods, send an empty JSON payload
                response = requests.request(method, url, json={}, timeout=5)
            
            # Check if the response indicates we have access (any response other than auth errors)
            if response.status_code in [401, 403]:
                return JsonResponse({
                    "success": False,
                    "error": f"Authentication failed with status code: {response.status_code}",
                    "message": f"Authentication failed with status code: {response.status_code}",
                    "status": response.status_code,
                    "endpoint": target_endpoint.get("name", ""),
                    "url": url
                }, status=200)  # Still return HTTP 200 even though the test failed
            else:
                return JsonResponse({
                    "success": True,
                    "status": response.status_code,
                    "endpoint": target_endpoint.get("name", ""),
                    "url": url,
                    "message": f"Endpoint is accessible (status code: {response.status_code})"
                }, status=200)
        except Exception as e:
            return JsonResponse({
                "success": False,
                "error": f"Request failed: {str(e)}",
                "endpoint": target_endpoint.get("name", ""),
                "url": url
            }, status=200)  # Still return HTTP 200 even though the test failed
            
    except json.JSONDecodeError:
        return JsonResponse({
            "success": False,
            "error": "Invalid JSON in request body",
            "status": 400
        }, status=400)
    except Exception as e:
        logging.error(f"Error in test_endpoint: {traceback.format_exc()}")
        return JsonResponse({
            "success": False,
            "error": f"Internal server error: {str(e)}",
            "status": 500
        }, status=500)


# STRIPE CHECKOUT STUFF

# Set Stripe API key
stripe.api_key = settings.STRIPE_SK

# Stripe webhook secret
WEBHOOK_SECRET = settings.STRIPE_WEBHOOK_SECRET

@csrf_exempt
def create_checkout_session(request):
    """
    Creates a Stripe Checkout Session for one-time token purchases.
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            product_id = data.get("product_id")
            user_id = data.get("user_id")
            print(user_id)

            if not user_id:
                return JsonResponse({"url": "https://limeblock.io/"})
            
            user = users_collection.find_one({'_id': ObjectId(user_id)})

            if not user:
                return JsonResponse({"error": "User not found"}, status=404)
            
            # Get existing Stripe customer ID if available
            stripe_customer_id = user.get('stripe_customer_id')
            
            # Product price mapping for one-time token purchases
            product_to_price_mapping = {
                "prod_SaalN81LV6nSbe": "price_1RfPZWCZciU921ANhPuM3x1M",  # $5 for 1M tokens
            }

            if product_id not in product_to_price_mapping:
                return JsonResponse({"error": "Invalid Product ID"}, status=400)

            # Session parameters for one-time payment
            session_params = {
                "payment_method_types": ["card"],
                "line_items": [
                    {
                        "price": product_to_price_mapping[product_id],
                        "quantity": 1,
                    }
                ],
                "mode": "payment",  # One-time payment mode (not subscription)
                "success_url": "https://limeblock.io/dashboard?payment=success",
                "cancel_url": "https://limeblock.io/dashboard?payment=cancelled",
                "metadata": {
                    "user_id": user_id,  # Attach user ID as metadata
                    "product_id": product_id,  # Attach product ID as metadata
                }
            }
            
            # Only add the customer ID if it exists
            if stripe_customer_id:
                session_params["customer"] = stripe_customer_id

            # Create a Stripe Checkout Session
            session = stripe.checkout.Session.create(**session_params)

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

    # Handle checkout.session.completed for one-time payments
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        handle_checkout_session(session)
    # Handle payment_intent.succeeded as backup
    elif event["type"] == "payment_intent.succeeded":
        payment_intent = event["data"]["object"]
        handle_payment_succeeded(payment_intent)

    return JsonResponse({"status": "success"}, status=200)


def handle_checkout_session(session):
    """
    Processes the checkout session for token purchases.
    """
    # Extract metadata from session
    user_id = session["metadata"].get("user_id")
    product_id = session["metadata"].get("product_id")
    
    if not user_id or not product_id:
        print("Missing required metadata in checkout session")
        return

    print(f"User {user_id} completed checkout for product {product_id}")
    
    try:
        # Get the user
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        
        if not user:
            print(f"User not found: {user_id}")
            return
        
        # Check if user already has a Stripe customer ID
        stripe_customer_id = user.get('stripe_customer_id')
        
        # If no Stripe customer ID exists, create one now
        if not stripe_customer_id:
            # Get the customer from the session
            customer_id = session.get('customer')
            
            if not customer_id:
                # Create a new customer if not available in session
                customer = stripe.Customer.create(
                    name=user.get('business_name', str(user['_id'])),
                    metadata={"user_id": str(user['_id'])}
                )
                stripe_customer_id = customer.id
            else:
                stripe_customer_id = customer_id
                
            # Save the customer ID to the user record
            users_collection.update_one(
                {'_id': ObjectId(user_id)},
                {'$set': {'stripe_customer_id': stripe_customer_id}}
            )
            print(f"Created Stripe customer {stripe_customer_id} for user {user_id}")
        
        # Credit tokens to user account
        credit_tokens_to_user(user_id, product_id)
    
    except Exception as e:
        print(f"Error processing checkout session: {e}")


def handle_payment_succeeded(payment_intent):
    """
    Handles payment_intent.succeeded event as a backup.
    """
    try:
        # Get metadata from payment intent
        user_id = payment_intent["metadata"].get("user_id")
        product_id = payment_intent["metadata"].get("product_id")
        
        if not user_id or not product_id:
            print("Missing required metadata in payment intent")
            return
        
        print(f"Payment succeeded for user {user_id}, product {product_id}")
        credit_tokens_to_user(user_id, product_id)
        
    except Exception as e:
        print(f"Error handling payment succeeded: {e}")


def credit_tokens_to_user(user_id, product_id):
    """
    Credits tokens to user's account based on the product purchased.
    """
    print(f"Crediting tokens for user {user_id} for product {product_id}")
    
    try:
        # Define token amounts for each product
        product_to_tokens = {
            "prod_SaalN81LV6nSbe": 1000000,  # 1 million tokens for $5
        }
        
        if product_id not in product_to_tokens:
            print(f"Unknown product ID: {product_id}")
            return
            
        tokens_to_add = product_to_tokens[product_id]
        
        # Get current user token balance
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if not user:
            print(f"User not found: {user_id}")
            return
            
        current_balance = user.get('tokens', 0)
        new_balance = current_balance + tokens_to_add
        
        # Update the user's token balance in the database
        result = users_collection.update_one(
            {'_id': ObjectId(user_id)}, 
            {
                '$set': {
                    'tokens': new_balance,
                    'last_paid': datetime.datetime.now()
                },
            }
        )
        
        if result.modified_count > 0:
            print(f"Successfully credited {tokens_to_add:,} tokens to user {user_id}")
            print(f"User balance: {current_balance:,} -> {new_balance:,} tokens")          
        else:
            print(f"Failed to update token balance for user {user_id}")
        
    except Exception as e:
        print(f"Failed to credit tokens to user: {e}")
        print(traceback.format_exc())


@csrf_exempt
def new_idea(request):
    try:
        # Get the MongoDB collection
        new_idea_collection = db['NewIdea']
        
        # Parse the JSON data from the request
        data = json.loads(request.body)
        
        # Create the document to insert
        document = {
            'message': data.get('message', ''),
            'email': data.get('email', ''),
            'budget': data.get('budget', ''),
            'companyUrl': data.get('companyUrl', ''),
            'created_at': datetime.datetime.now(),
            'processed': False,
            'status': 'pending'
        }
        
        # Insert the document into MongoDB
        result = new_idea_collection.insert_one(document)
        
        # Return success response
        return JsonResponse({
            'success': True,
            'message': 'Idea submitted successfully',
            'id': str(result.inserted_id)
        }, status=201)
        
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON data'
        }, status=400)
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)