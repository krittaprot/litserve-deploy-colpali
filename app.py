import litserve as ls
import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from io import BytesIO
from PIL import Image
import base64
import os

#Enable hf_transfer for faster download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" # optional setting for faster dataset downloads

class ColPaliLitAPI(ls.LitAPI):
    def setup(self, device):
        # Load the model and processor
        self.model = ColPali.from_pretrained(
            "vidore/colpali-v1.3",
            torch_dtype=torch.bfloat16,
            device_map='mps',  # Use "cuda:0" for GPU, "cpu" for CPU, or "mps" for Apple Silicon
        ).eval()
        
        self.processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3")

    def decode_request(self, request):
        # Decode the request to extract either text or image
        if "images" in request:
            # Assuming the images are sent as a list of base64 encoded strings
            images = request["images"]
            return {"images": images}
        elif "texts" in request:
            return {"texts": request["texts"]}
        else:
            raise ValueError("Request must contain either 'images' or 'texts'")

    def predict(self, inputs):
        # Process the inputs and get embeddings
        if "images" in inputs:
            # Convert base64 strings to PIL Images
            from io import BytesIO
            images = [Image.open(BytesIO(base64.b64decode(image_data))) for image_data in inputs["images"]]
            batch_images = self.processor.process_images(images).to(self.model.device)
            with torch.no_grad():
                image_embeddings = self.model(**batch_images)
            return {"embeddings": image_embeddings.float().cpu().numpy().tolist()}
        elif "texts" in inputs:
            batch_queries = self.processor.process_queries(inputs["texts"]).to(self.model.device)
            with torch.no_grad():
                query_embeddings = self.model(**batch_queries)
            return {"embeddings": query_embeddings.float().cpu().numpy().tolist()}
        else:
            raise ValueError("Inputs must contain either 'images' or 'texts'")

    def encode_response(self, output):
        # Encode the response to return the embeddings
        return {"embeddings": output["embeddings"]}
    
if __name__ == "__main__":
    api = ColPaliLitAPI()
    server = ls.LitServer(api, accelerator="cpu")
    server.run(port=8000)