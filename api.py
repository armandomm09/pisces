from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from classification import EmbeddingClassifier, YOLOInference, SegmentationInference
import os
import json
import uuid
app = FastAPI()

detector = YOLOInference("models/bb_model.ts")
classifier = EmbeddingClassifier("models/class_model.ts", "database.pt", detector, device="cpu")
segmentator = SegmentationInference("models/segmentation_21_08_2023.ts")
with open("labels/vernacular_names.json", 'r') as file:
    data = json.load(file)
    
def get_fish_nickname(name):
    return data[name]

@app.get("/")
async def default():
    return {"Hello": "World"}

@app.post("/image")
async def image(file: UploadFile = File(...)):
    try:
        # Lee el contenido del archivo subido
        contents = file.file.read()
        
        image_uuid = str(uuid.uuid4())
        image_name = f"{image_uuid}.jpg"
        file_path = f"uploads/{image_name}"
        
        
        # Escribe el archivo en el sistema de archivos
        with open(file_path, "wb") as f:
            f.write(contents)

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        file.file.close()
        
    
    out_image_path = f"downloads/{image_name}"
    fish_info = classifier.classify_image(file_path, out_image_path, )
    segmentator.segmentate_img(out_image_path, out_image_path, (255, 0, 0, 255))
    last_img_path = file_path
    print("File saved at: ", file_path)
    return {"message": f"Successfully uploaded {image_name} to {file_path}", "filename": image_name, "fish_info": fish_info, "nickname": get_fish_nickname(fish_info[0]["name"])}
    
@app.get("/fetch_image/{image_name}")
async def get_img(image_name):
    return FileResponse(f"downloads/{image_name}", media_type="image/jpg")
       