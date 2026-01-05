import os, json, time, ijson
from decimal import Decimal
from openai import OpenAI

# removed api key 
# os.environ['OPENAI_API_KEY'] = 

# intermediary input/output files from the prompt generation process that we have not included, all data used for plotting and stats is in processed_data_cache.pkl, complete data including captions is in "out_final_4_k_limited_with_scores.json" and "OF4_gpt4o_updated.json"
INPUT_FILE = "OF4_gpt4o_prep.json" 
OUTPUT_FILE = "OF4_gpt4o.json" 
SAVE_EVERY = 100
OPENAI_MODEL = "gpt-4o-mini"

client = OpenAI()

class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)

def gpt_once(prompt: str, retries: int = 3, wait: int = 60) -> str:
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model = OPENAI_MODEL,
                messages = [
                    {"role": "system", "content": "You are a helpful assistant who follows instructions exactly."},
                    {"role": "user", "content": prompt},
                ],
                timeout = 90
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[{attempt}/{retries}] OpenAI error: {e}")
            if attempt == retries:
                print("giving up on this image")
                return ""
            time.sleep(wait)

with open(INPUT_FILE, "r") as fin, open(OUTPUT_FILE, "w") as fout:
    parser = ijson.kvitems(fin, "")
    fout.write("{\n")
    first_out = True
    processed = 0

    for fname, img in parser:
        base_caption = img["generated_caption_4"]

        concise_prompt = (
            "Describe this image and don't introduce any emotional information. "
            "Just describe what's there. Be as concise as possible.\n\n"
            f"{base_caption}"
        )
        p200_prompt = (
            "Describe this image and don't introduce any emotional information. "
            "Just describe what's there. Don't exceed 200 characters.\n\n"
            f"{base_caption}"
        )

        concise_txt = gpt_once(concise_prompt)
        p200_txt = gpt_once(p200_prompt)

        img["gpt4o_concise"] = concise_txt
        img["gpt4o_concise_length"] = len(concise_txt)
        img["gpt4o_200char"] = p200_txt
        img["gpt4o_200char_length"] = len(p200_txt)

        if not first_out:
            fout.write(",\n")
        first_out = False

        json.dump(fname, fout)
        fout.write(": ")
        json.dump(img, fout, cls = DecimalEncoder)

        processed += 1

        if processed % SAVE_EVERY == 0:
            fout.flush()
            print(f"{processed} images processed")

    fout.write("\n}\n")

print(f"{processed} images written to {OUTPUT_FILE}")
