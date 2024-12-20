# Project: Fish Detection and Classification

This project combines computer vision, machine learning, and WhatsApp integration to build a robust system capable of detecting, classifying, and segmenting images of fish sent via the WhatsApp application. Here is a technical explanation of the components and their functioning.

---

### Project Architecture

#### **Frontend Interaction: WhatsApp Web API**
The project uses the `whatsapp-web.js` library to interact with WhatsApp messages and multimedia files. This component includes:
- Local authentication to save session credentials.
- Image processing of images sent to the chat.
- Communication with the FastAPI backend for image classification.


#### **Backend: FastAPI for Image Processing**
A FastAPI server handles requests related to images, including classification, segmentation, and result generation.
- **Classification**: Identifies fish species using the `EmbeddingClassifier` with its cientifical name.
- **Segmentation**: Highlights regions of interest in the image using `SegmentationInference`.
- **Model Integration**: Pre-trained models (YOLOInference, EmbeddingClassifier, and segmentation) are stored on the server to ensure accuracy in results.
- **Custom Nicknames**: Based on a `labels/vernacular_names.json` file, it provides common names for detected species.

*Suggested Image:*  



---

### **File Storage and Management**

- **uploads/ directory**: Stores original images sent by users.
- **downloads/ directory**: Stores processed images with segmentation and other visual annotations.

*Suggested Image:*  
Representation of the system folders (uploads/ and downloads/).

---

### **Image Processing Pipeline**

- Images are sent as files through WhatsApp.
- The FastAPI API processes the images in several steps:
  1. Initial fish classification.
  2. Segmentation of the relevant region.
  3. Generation of results, including nicknames.
- The processed image is sent back through WhatsApp, along with a message containing the fish’s name and nickname.

*Suggested Image:*  
Flowchart illustrating the process from image reception to the sending of results.

---

### **Communication between WhatsApp and FastAPI**

The WhatsApp client uses `fetch` to make POST and GET requests to the FastAPI server:
- **POST**: Sends the image for processing.
- **GET**: Retrieves the processed image.

---

### **Technologies Used**

- **FastAPI**: Used to build the API backend.
- **whatsapp-web.js**: Used for WhatsApp interaction.
- **YOLO**: Used for object detection.
- **Classification and Segmentation**: Custom models specifically trained for fish detection.
- **Python**: The main language for backend and image processing.
- **JavaScript**: The main language for WhatsApp integration.

---

### **Expected Results**

The system processes images sent to WhatsApp, classifies the fish species, generates personalized nicknames, and applies visual segmentation. The results are returned to the user as a WhatsApp message enriched with text and images.

---

Feel free to adjust any sections if you need more specific details.