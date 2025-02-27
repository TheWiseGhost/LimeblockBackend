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
blocks_collection = db['Blocks']


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
            return JsonResponse(
                {"success": "User added successfully", "id": str(result.inserted_id)},
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

