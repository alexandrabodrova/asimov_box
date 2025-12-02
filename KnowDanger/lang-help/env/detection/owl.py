import requests
from PIL import Image
import torch
import time

from transformers import OwlViTProcessor, OwlViTForObjectDetection


class OWL():

    def __init__(self):
        processor = OwlViTProcessor.from_pretrained(
            "google/owlvit-base-patch32"
        )
        model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32"
        )
        # processor = OwlViTProcessor.from_pretrained(
        #     "google/owlvit-large-patch14"
        # )
        # model = OwlViTForObjectDetection.from_pretrained(
        #     "google/owlvit-large-patch14"
        # )

        # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        # image = Image.open(requests.get(url, stream=True).raw)
        image_path = 'env/test/tmp.jpg'
        image = Image.open(image_path)
        # texts = [["a photo of a cat", "a photo of a dog"]]
        texts = [[
            "a yellow bowl",
            'a green bowl',
            'a blue bowl',
            "a blue cube",
            "a green cube",
            'a yellow cube',
        ]]
        inputs = processor(text=texts, images=image, return_tensors="pt")
        s1 = time.time()
        outputs = model(**inputs)
        print('Inference time:', time.time() - s1)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes
        )

        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i][
            "scores"], results[i]["labels"]

        score_threshold = 0.05
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            if score >= score_threshold:
                print(
                    f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}"
                )
        # Detected a photo of a cat with confidence 0.707 at location [324.97, 20.44, 640.58, 373.29]
        # Detected a photo of a cat with confidence 0.717 at location [1.46, 55.26, 315.55, 472.17]

        # visualize boxes with scores on the image
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        for box, score, label in zip(boxes, scores, labels):
            if score >= score_threshold:
                draw.rectangle(box.tolist(), outline="red")
                draw.text(
                    box[:2],
                    f"{text[label]} {round(score.item(), 3)}",
                    fill="orange",
                )
        image.show()


if __name__ == "__main__":
    owl = OWL()
