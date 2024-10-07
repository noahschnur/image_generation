import os
import openai
from dotenv import load_dotenv

from diffusers import StableDiffusionPipeline
from PIL import Image
import torch

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")

def enhance_prompt(prompt):
  """
  Enhances the user prompt using OpenAI's GPT-3 model.
  """
  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-0125",
      messages=[
          {"role": "system", "content": "You are a helpful and informative AI assistant. You enhance and edit user prompts to be used in an image generating tool. Your responses are 100 words or less, and adhere to PG-13 guidelines. Enhance the following prompt:"},
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
  image_folder = "Generated_images"
  os.makedirs(image_folder, exist_ok=True)  # Ensures folder creation

  # Save the image within the "Generated_images" folder
  full_path = os.path.join(image_folder, filename)
  image.save(full_path)
  return image


if __name__ == "__main__":
  prompt = input("Enter a prompt: ")
  enhanced_prompt = enhance_prompt(prompt)
  print(f"Enhanced Prompt: {enhanced_prompt}")

  # Keep track of the image index
  image_index = 0

  while True:
    generated_image = generate_image(enhanced_prompt, image_index)
    generated_image.show()
    image_index += 1  # Increment index for the next image

    prompt = input("Enter another prompt (or 'q' to quit): ")
    if prompt.lower() == 'q':
      break

  print("Image complete!")