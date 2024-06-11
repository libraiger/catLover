from PIL import Image, ImageDraw, ImageFont

# Load the uploaded kitten image
kitten_image_path = "img.png"
kitten_image = Image.open(kitten_image_path)

# Create a pair of sunglasses to put on the kitten
sunglasses_width = 200
sunglasses_height = 50
sunglasses = Image.new('RGBA', (sunglasses_width, sunglasses_height), (0, 0, 0, 255))

# Add sunglasses to the kitten image
kitten_with_sunglasses = kitten_image.copy()
draw = ImageDraw.Draw(kitten_with_sunglasses)
sunglasses_position = (170, 140)  # Approximate position for the sunglasses on the kitten's face

# Overlay the sunglasses onto the kitten image
kitten_with_sunglasses.paste(sunglasses, sunglasses_position, sunglasses)

# Save and display the result
output_path = "kitten_with_sunglasses.png"
kitten_with_sunglasses.save(output_path)