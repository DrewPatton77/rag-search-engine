import argparse
import mimetypes
from call_llm import llm_image
import base64

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal CLI")
    parser.add_argument("--image", type=str, nargs="?", help="The file path for the image to be embedded")
    parser.add_argument("--query", type=str, nargs="?", help="The query to be embedded")

    args = parser.parse_args()

    image = args.image
    query = args.query

    mime, _ = mimetypes.guess_type(image)
    mime = mime or "image/jpeg"

    with open(image, "rb") as f:
        img = f.read()

    data_url = f"data:{mime};base64,{base64.b64encode(img).decode()}"

    response = "User Safety: safe"
    while response == "User Safety: safe":
        response = llm_image(data_url, query)

    print("Rewritten query:")
    print(response)


if __name__ == "__main__":
    main()
