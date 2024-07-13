import re
import json

# Read the improperly formatted file
with open('imagenet2_classes.json', 'r') as f:
    data = f.read()

# Use regex to replace single quotes with double quotes and add double quotes around keys
data = re.sub(r"(\d+): '([^']+)'", r'"\1": "\2"', data)

# Add curly braces to make it a valid JSON object
data = "{" + data + "}"

# Write the properly formatted JSON data to a new file
with open('imagenet_classes.json', 'w') as f:
    f.write(data)

print("Conversion to JSON format completed.")
