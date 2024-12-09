import os
import pickle

# Define the function to get the product URL
def get_product_page_url(image_name):
    # Replace this with your actual logic to generate or fetch the product URL
    return f"https://example.com/product/{image_name}"

# Prepare the image paths and URLs
img_files = []

# Assuming all images are in 'fashion_small/images' directory
for fashion_image in os.listdir('fashion_small/images'):
    image_path = os.path.join('fashion_small/images', fashion_image)
    product_page_url = get_product_page_url(fashion_image)
    img_files.append({"image_path": image_path, "product_url": product_page_url})

# Save the list of dictionaries to pickle
pickle.dump(img_files, open("img_files.pkl", "wb"))
print("Image files pickle has been generated successfully.")