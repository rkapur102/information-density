import os
import json
import time
from openai import OpenAI
import base64

# removed api key 
# os.environ['OPENAI_API_KEY'] = 

client = OpenAI()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_caption_with_image(prompt, image_path, retries = 3, wait_time = 60):
    attempt = 0
    while attempt < retries:
        try:
            base64_image = encode_image(image_path)
            response = client.chat.completions.create(
                model = "gpt-4o-mini",
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            attempt += 1
            print(f"Error occurred: {e}. Attempt {attempt} of {retries}. Waiting for {wait_time} seconds before retrying.")
            time.sleep(wait_time)
            if attempt == retries:
                print("Max retries reached. Skipping this request.")
                return None

# intermediary input/output files from the prompt generation process that we have not included
input_file = './updated_images_and_captions.json'
output_file = './updated_images_and_captions_with_sole_image_input.json'

with open(input_file, 'r') as f:
    image_data = json.load(f)

updated_images = {}
num = 0

for img_file_name, img_info in list(image_data.items())[0:]:
    print(num)

    prompt_image_description = "Describe this image and don't introduce any emotional information. Just describe what's there."
    image_path = img_info['image_path']

    img_info['generated_caption_4'] = generate_caption_with_image(prompt_image_description, image_path = image_path)

    updated_images[img_file_name] = img_info
    num += 1

    if num % 100 == 0:
        with open(output_file, 'w') as f:
            json.dump(updated_images, f, indent = 4)
        print(f"Progress saved after {num} images.")

with open(output_file, 'w') as f:
    json.dump(updated_images, f, indent = 4)

print(f"Final updated captions saved to {output_file}")