import cv2
import numpy as np
import time
from collections import defaultdict
import logging

class ModelEnsemble:
    # --- CHANGED: Added 'classes_path' to the constructor ---
    def __init__(self, configs, classes_path):
        self.models = {}
        self.performance_history = defaultdict(list)
        self.model_weights = {}
        
        # Load all models
        for model_name, (cfg_path, weights_path) in configs.items():
            self.models[model_name] = self.load_model(cfg_path, weights_path)
            self.model_weights[model_name] = 1.0  # Initial equal weight
        
        # --- CHANGED: Now uses the provided path to open the classes file ---
        try:
            with open(classes_path, 'r') as f:
                self.classes = f.read().strip().split('\n')
        except FileNotFoundError:
            print(f"❌ FATAL: Classes file not found at {classes_path}")
            self.classes = [] # Set to empty list to avoid further errors

        self.logger = logging.getLogger('ModelEnsemble')
    
    def load_model(self, cfg_path, weights_path):
        """Load YOLO model"""
        net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net
    
    def get_output_layers(self, net):
        """Get output layer names"""
        layer_names = net.getLayerNames()
        try:
            # For OpenCV 4.x and later
            return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except TypeError:
            # Fallback for older OpenCV versions
             return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def detect_single_model(self, model, image, confidence_threshold=0.5):
        """Perform detection with a single model"""
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        model.setInput(blob)
        outputs = model.forward(self.get_output_layers(model))
        
        return self.process_outputs(outputs, image.shape, confidence_threshold)
    
    def process_outputs(self, outputs, image_shape, confidence_threshold):
        """Process YOLO outputs to get detections"""
        h, w = image_shape[:2]
        boxes, confidences, class_ids = [], [], []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    width = int(detection[2] * w)
                    height = int(detection[3] * h)
                    
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    
                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        return boxes, confidences, class_ids
    
    def weighted_voting(self, all_detections, iou_threshold=0.5):
        """Combine detections using weighted voting based on confidence"""
        all_boxes = []
        all_scores = []
        all_labels = []
        all_weights = []
        
        # Collect all detections with their model weights
        for model_name, (boxes, confidences, class_ids) in all_detections.items():
            weight = self.model_weights[model_name]
            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                all_boxes.append(box)
                all_scores.append(confidence * weight)  # Weighted confidence
                all_labels.append(class_id)
                all_weights.append(weight)
        
        if not all_boxes:
            return [], [], []
        
        # Apply non-maximum suppression with weighted scores
        return self.weighted_nms(all_boxes, all_scores, all_labels, iou_threshold)
    
    def weighted_nms(self, boxes, scores, labels, iou_threshold):
        """Non-maximum suppression that considers weighted scores"""
        if len(boxes) == 0:
            return [], [], []
        
        boxes = np.array(boxes)
        scores = np.array(scores)
        labels = np.array(labels)
        
        # Convert to x1, y1, x2, y2 format
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        final_boxes = boxes[keep].tolist()
        final_scores = scores[keep].tolist()
        final_labels = labels[keep].tolist()
        
        return final_boxes, final_scores, final_labels
    
    def adaptive_model_selection(self, image):
        """Select best model based on scene complexity"""
        # Simple scene complexity analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        complexity = np.std(gray)  # Use standard deviation as complexity measure
        
        # Update weights based on complexity
        if complexity < 30:  # Simple scene
            self.model_weights['yolov4-tiny'] = 0.8  # Prefer faster model
            self.model_weights['yolov3'] = 0.2
        else:  # Complex scene
            self.model_weights['yolov3'] = 0.8  # Prefer more accurate model
            self.model_weights['yolov4-tiny'] = 0.2
        
        self.logger.info(f"Scene complexity: {complexity:.2f}, Weights: {self.model_weights}")
    
    def update_model_weights(self, model_name, performance_metric):
        """Dynamically update model weights based on performance"""
        self.performance_history[model_name].append(performance_metric)
        
        # Use average of last 10 performances
        recent_performance = np.mean(self.performance_history[model_name][-10:])
        
        # Normalize weights across all models
        total_performance = sum(np.mean(self.performance_history[name][-10:]) for name in self.models if self.performance_history[name])
        if total_performance > 0:
            self.model_weights[model_name] = recent_performance / total_performance
        
        self.logger.info(f"Updated weights: {self.model_weights}")
    
    def ensemble_detect(self, image, confidence_threshold=0.5):
        """Main ensemble detection method"""
        if not self.classes:
            print("❌ Cannot perform detection, classes not loaded.")
            return [], [], []

        start_time = time.time()
        
        # Adaptive model selection based on scene
        self.adaptive_model_selection(image)
        
        all_detections = {}
        individual_performances = {}
        
        # Get detections from all models
        for model_name, model in self.models.items():
            model_start = time.time()
            boxes, confidences, class_ids = self.detect_single_model(
                model, image, confidence_threshold
            )
            model_time = time.time() - model_start
            
            all_detections[model_name] = (boxes, confidences, class_ids)
            
            # Calculate performance metric (higher is better)
            performance = len(boxes) / (model_time + 1e-5)  # Detections per second
            individual_performances[model_name] = performance
            
            self.logger.info(f"{model_name}: {len(boxes)} detections in {model_time:.3f}s")
        
        # Update weights based on individual performance
        for model_name, performance in individual_performances.items():
            self.update_model_weights(model_name, performance)
        
        # Combine detections using weighted voting
        final_boxes, final_scores, final_labels = self.weighted_voting(all_detections)
        
        total_time = time.time() - start_time
        self.logger.info(f"Ensemble: {len(final_boxes)} final detections in {total_time:.3f}s")
        
        return final_boxes, final_scores, final_labels

    def visualize_detections(self, image, boxes, scores, labels, output_path=None):
        """Visualize detections on image"""
        result_image = image.copy()
        
        for box, score, label_id in zip(boxes, scores, labels):
            if label_id >= len(self.classes):
                print(f"⚠️ Warning: Invalid label ID {label_id} detected. Skipping.")
                continue

            x, y, w, h = box
            label_text = f"{self.classes[label_id]}: {score:.2f}"
            
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(result_image, (x, y - label_height - 10), 
                          (x + label_width, y), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(result_image, label_text, (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, result_image)
        
        return result_image