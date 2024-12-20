from classification import EmbeddingClassifier, YOLOInference, SegmentationInference
import uuid

detector = YOLOInference("models/bb_model.ts")
classifier = EmbeddingClassifier("models/class_model.ts", "database.pt", detector, device="cpu")
segmentator = SegmentationInference("models/segmentation_21_08_2023.ts")

in_path = "images/bass.jpg"
image_uuid = str(uuid.uuid4())
out_path = f"downloads/{image_uuid}.jpg"
print(out_path)
fish_info = classifier.classify_image(in_path, out_path, )
segmentator.segmentate_img(out_path, out_path, (255, 0, 0, 255))

print(fish_info)