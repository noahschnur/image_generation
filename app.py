from flask import Flask, render_template, request
import os
import openai
from dotenv import load_dotenv

from diffusers import StableDiffusionPipeline
from PIL import Image
import torch

app = Flask(__name__, static_folder='static')

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")

def enhance_prompt(prompt):
  """
  Enhances the user prompt using OpenAI's GPT-3 model.
  """
  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-0125",
      messages=[
          {"role": "system", "content": "You are a helpful and informative AI assistant. You enhance and edit user prompts to be used in an image generating tool. Your responses are 100 words or less and you remove the name of any celebrity, professional athlete, and politician. You adhere to the MPAA rating system in the following ways: PG-13 guidelines for vilolence and language, and PG for all other topics. Enhance the following prompt:"},
          {"role": "user", "content": prompt}
      ],
      max_tokens=1024,
      n=1,
      stop=None,
      temperature=0.5
  )
  return response.choices[0].message["content"]


def generate_image(prompt, index=0):
  """
  Generates an image using Stable Diffusion based on the provided prompt.
  Optionally takes an index for naming the generated image.
  """
  pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
  image = pipeline(prompt).images[0]

  # Generate the filename with incremental numbering
  filename = f"generated_image_{index}.png"

  # Create the "Generated_images" folder if it doesn't exist
  image_folder = "static/Generated_images"
  os.makedirs(image_folder, exist_ok=True)  # Ensures folder creation

  # Save the image within the "Generated_images" folder
  full_path = os.path.join(image_folder, filename)
  image.save(full_path)
  return image


@app.route("/", methods=["GET", "POST"])
def generate_image_view():
    if request.method == "POST":
        prompt = request.form["prompt"]
        enhanced_prompt = enhance_prompt(prompt)
        
        # Get the count of existing images to use as the index
        image_folder = "static/Generated_images"
        existing_images = len([f for f in os.listdir(image_folder) if f.endswith('.png')])
        
        # Generate the image with the current index
        generated_image = generate_image(enhanced_prompt, index=existing_images)
        
        # Construct the correct filename
        image_filename = f"generated_image_{existing_images}.png"
        
        # Construct the correct path for the template
        image_path = f"Generated_images/{image_filename}"
        
        return render_template("index.html", prompt=prompt, enhanced_prompt=enhanced_prompt, image_path=image_path)
    else:
        return render_template("index.html")

if __name__ == "__main__":
  app.run(debug=True)