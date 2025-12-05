import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image

# ==========================================
# PART 1: The "Kaggle-Trained" Classification Model
# ==========================================
class DefectClassifier:
    def __init__(self):
        # In a real scenario, you would load your custom trained model here:
        # self.model = tf.keras.models.load_model('neu_det_steel_defects.h5')
        
        # For this demo, we use a pre-trained MobileNetV2 as a placeholder.
        # We simulate the output classes relevant to machine parts.
        self.model = MobileNetV2(weights='imagenet', include_top=True)
        self.defect_classes = [
            'Scratches', 'Cracks', 'Corrosion', 'Dents', 'Surface Anomalies', 'No Defect'
        ]

    def predict_defect(self, img_path):
        """
        Loads an image and predicts the defect type.
        """
        try:
            # Preprocess image for the model
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Get prediction (Simulation logic for demo purposes)
            # In a real app: preds = self.model.predict(img_array)
            # Here we just simulate a random defect detection for demonstration
            import random
            detected_defect = random.choice(self.defect_classes)
            confidence = random.uniform(0.85, 0.99)
            
            return {
                "filename": os.path.basename(img_path),
                "defect_type": detected_defect,
                "confidence": f"{confidence:.2%}",
                "action_required": "Inspect manually" if detected_defect != 'No Defect' else "None"
            }
        except Exception as e:
            return {"error": str(e)}

# ==========================================
# PART 2: Google ADK Multi-Agent Structure
# ==========================================

# NOTE: This uses a simplified class structure to mimic the ADK behavior 
# so it runs without needing the specific private/preview ADK keys.

class Agent:
    """Base Agent class simulating Google ADK Agent behavior."""
    def __init__(self, name, role, tools=[]):
        self.name = name
        self.role = role
        self.tools = tools

    def run(self, input_data):
        print(f"\n[Agent: {self.name}] Received input: {input_data}")
        print(f"[Agent: {self.name}] Reasoning: {self.role}")
        
        # If this agent has tools, use them
        if self.tools:
            for tool in self.tools:
                print(f"[Agent: {self.name}] invoking tool: {tool.__name__}...")
                result = tool(input_data)
                return result
        return "Task delegated."

# --- Define the Tools ---
classifier = DefectClassifier()

def inspect_machine_part_tool(image_path):
    """Tool that uses the classifier to check for defects."""
    return classifier.predict_defect(image_path)

# --- Define the Agents ---

# Agent 1: The Manager (Routes the request)
manager_agent = Agent(
    name="Production_Manager",
    role="Decide if the image needs inspection and assign to Inspector."
)

# Agent 2: The Inspector (Performs the analysis)
inspector_agent = Agent(
    name="Quality_Inspector",
    role="Analyze machine part image for cracks, corrosion, or dents.",
    tools=[inspect_machine_part_tool]
)

# ==========================================
# PART 3: Execution Flow
# ==========================================
def main():
    # 1. Setup Input Folder
    input_folder = "machine_images_input"
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"Created folder '{input_folder}'. Please put images there.")
        # Create a dummy image for the demo
        dummy_img = Image.new('RGB', (224, 224), color = (73, 109, 137))
        dummy_img.save(os.path.join(input_folder, "gear_sample_01.jpg"))

    # 2. Process images
    print("--- Starting Multi-Agent Defect Detection System ---")
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            
            # Step A: Manager receives the file
            manager_decision = manager_agent.run(f"New image received: {filename}")
            
            # Step B: Manager delegates to Inspector
            result = inspector_agent.run(image_path)
            
            # Step C: Output Result
            print(f"\n✅ FINAL REPORT for {filename}:")
            print(result)
            print("-" * 30)

if __name__ == "__main__":
    main()