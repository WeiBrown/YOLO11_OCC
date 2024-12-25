from ultralytics import YOLO
try:
    from .ultralytics import YOLO
except:
    pass

# Load a model
# model = YOLO("yolo11n.yaml")  # build a new model from YAML
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11x.yaml").load("yolo11x.pt")  # build from YAML and transfer weights
model = YOLO("yolo11x-occ.yaml").load("yolo11x.pt")  # build from YAML and transfer weights
# model = YOLO("yolo11x.yaml").load("yolo11x.pt")  # build from YAML and transfer weights

# Train the model
# results = model.train(data="kitti_mini.yaml", epochs=100, imgsz=1280,device=2,batch=2,workers=0,augment=False)
results = model.train(data="kitti_mini.yaml", epochs=100, imgsz=1280,device=2,augment=False)