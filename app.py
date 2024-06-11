from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # Load the environment variables from .env file

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))


response = client.images.edit(
  model="dall-e-2",
  image=open("maga.png", "rb"),
  mask=open("maga_mask.png", "rb"),
  prompt="a cat wear the sunglasses on it's eyes and hat on it's head",
  n=10,
  size="512x512"
)

# Get all the URLs of the generated images
image_urls = [image.url for image in response.data]
print(image_urls)