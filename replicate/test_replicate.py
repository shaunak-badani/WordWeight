from dotenv import load_dotenv
import os
load_dotenv()
import replicate

import os
print("Key : ", os.environ.get("REPLICATE_API_TOKEN"))

client = replicate.Client(api_token = os.environ.get("REPLICATE_API_TOKEN"))

with open("./masked_image.png", "rb") as mask_file:
    output = client.run(
        "shaunak-badani/wordweight:1cfd38753b1f35fd2edd9a949dcd655dd77b9b2f6840adb8100fd6f3ac298183",
        input={
            "prompt": "a futuristic cityscape",
            "mode": "generate",
            "mask_path": mask_file
        }
    )
    print(output)