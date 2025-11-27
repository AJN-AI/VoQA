import base64
import requests

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

# Here we take the api template of a certain website as an example
def post_data(API_KEY, model_name, image_path, prompt):
    # print(f"image_path: {image_path}, prompt: {prompt}.")

    url_name = "https://XXXX/"
    API_URL = url_name + "XXXX"
    # API_KEY = "XXXX"

    base64_image = encode_image(image_path)  # encoding local image

    # Create the request data payload, including the required model and message content
    payload = {
        "model": f"{model_name}", # For the specified multimodal AI models to be used, in addition to gpt-4o, the claude-3-5-sonnet series is also recommended
        "messages": [
            {
                "role": "system",  # System role information can be empty
                "content": "",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            # Local images encoded in Base64
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                ],
            },
        ],
        "temperature": 0,
        "user": "XXX",  # User identification for sending requests
    }

    # Define the HTTP request header, including the content type and authentication information
    headers = {
        "Content-Type": "application/json",  # Set the content type to JSON
        "Authorization": f"XXx {API_KEY}",  # Use f-string to dynamically insert API_KEY for authentication
        "User-Agent": f"XXX/1.0.0 ({url_name})",  # Custom User-Agent for identifying client information
    }

    # Send a POST request, pass the request data and header information into the API, and obtain the response
    response = requests.post(API_URL, headers=headers, json=payload)

    # Output the response content of the API
    # print(response.text)
    # You should modify it according to the actual situation
    response_json = response.json()
    return response_json["choices"][0]["message"]["content"], response.text

