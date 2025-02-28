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

client = MongoClient(f'{settings.MONGO_URI}')
db = client['Limeblock']
users_collection = db['Users']
frontend_collection = db['Frontend']
frontend_endpoints_collection = db['Frontend_Endpoints']
backend_collection = db['Backend']
backend_endpoints_collection = db['Backend_Endpoints']

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

        # Prepare user data to insert
        user_data = {
            "business_name": business_name,
            "password": password,
            "emails": emails,
            "created_at": datetime.datetime.today(),
            "plan": "free",
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
            }

            backend = {
                "user_id": str(result.inserted_id),
                "url": "",
                "folders": [],
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
                    "folders": frontend.get('folders', [])
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
    