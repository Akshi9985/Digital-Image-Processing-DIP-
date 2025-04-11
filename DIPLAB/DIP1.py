from PIL import Image
import os

# Open the image
img = Image.open("DIPLAB/dip.img.jpg")
# Create the directory to save images if it doesn't exist
os.makedirs("D:/DIPLAB/SaveImages", exist_ok=True)

# Save the image in different formats
img.save("D:/DIPLAB/SaveImages/dip_saved(jpg).jpg")
print("Image saved successfully!")

img.save("D:/DIPLAB/SaveImages/dip_saved(png).png", format="PNG")
print("Image saved successfully in PNG format!")

img.save("D:/DIPLAB/SaveImages/dip_saved(gif).gif", format="GIF")
print("Image saved successfully in GIF format!")

img.save("D:/DIPLAB/SaveImages/dip_saved(tiff).tiff", format="TIFF")
print("Image saved successfully in TIFF format!")

img.save("D:/DIPLAB/SaveImages/dip_saved(webp).webp", format="WEBP")
print("Image saved successfully in WEBP format!")

# List of saved image files
image_files = [
    "D:/DIPLAB/SaveImages/dip_saved(jpg).jpg", 
    "D:/DIPLAB/SaveImages/dip_saved(png).png", 
    "D:/DIPLAB/SaveImages/dip_saved(gif).gif", 
    "D:/DIPLAB/SaveImages/dip_saved(tiff).tiff", 
    "D:/DIPLAB/SaveImages/dip_saved(webp).webp", 
]

# Print properties of each image
for image_file in image_files:
    img = Image.open(image_file)
    print(f"\nImage Properties for {image_file}:")
    print(f"Format: {img.format}")
    print(f"Mode: {img.mode}") 
    print(f"Size: {img.size}") 
    print(f"Width: {img.width}")  
    print(f"Height: {img.height}")  
    print(f"Info: {img.info}")