import os
import random
import json
import time
from openai import OpenAI

# removed api key 
# os.environ['OPENAI_API_KEY'] = 

client = OpenAI()

def generate_caption_gpt4o(prompt, retries = 3, wait_time = 60):
    attempt = 0
    while attempt < retries:
        try:
            completion = client.chat.completions.create(
                model = "gpt-4o-mini",
                messages = [
                    {"role": "system", "content": "You are a helpful assistant who follows instructions exactly."},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            attempt += 1
            print(f"Error occurred: {e}. Attempt {attempt} of {retries}. Waiting for {wait_time} seconds before retrying.")
            time.sleep(wait_time)
            if attempt == retries:
                print("Max retries reached. Skipping this request.")
                return None

# intermediary input/output files from the prompt generation process that we have not included, all data used for plotting and stats is in processed_data_cache.pkl, complete data including captions is in "out_final_4_k_limited_with_scores.json" and "OF4_gpt4o_updated.json"
input_file = './selected_images_and_captions_with_primary_and_all_categories.json'
output_file = './updated_images_and_captions.json'

with open(input_file, 'r') as f:
    image_data = json.load(f)

updated_images = {}

for img_file_name, img_info in list(image_data.items()):
    print(num)
    captions = img_info['captions']

    prompt_all_captions = (
        f"Given these 5 descriptions, generate one longer, final description that combines all information in the individual descriptions. "
        f"Do not augment the description with any emotional or made-up information. Only output the longer description and nothing else.\n\n"
        f"{captions[0]}\n{captions[1]}\n{captions[2]}\n{captions[3]}\n{captions[4]}"
    )
    img_info['generated_caption_2'] = generate_caption_gpt4o(prompt_all_captions)

    random_caption = random.choice(captions)
    prompt_random_caption = (
        f"Given this description, generate one longer description that expresses the same information as in the original description but in a more verbose way. "
        f"In other words, use more words but say the same thing as given. Do not augment the description with any emotional or made-up information. "
        f"Only output the longer description and nothing else.\n\n"
        f"{random_caption}"
    )
    img_info['generated_caption_3'] = generate_caption_gpt4o(prompt_random_caption)

    updated_images[img_file_name] = img_info
    num += 1

    if num % 100 == 0:
        with open(output_file, 'w') as f:
            json.dump(updated_images, f, indent = 4)
        print(f"Progress saved after {num} images.")

with open(output_file, 'w') as f:
    json.dump(updated_images, f, indent = 4)

print(f"Final updated captions saved to {output_file}")
