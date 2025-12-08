import numpy as np
from collections import defaultdict, deque

class ObjectTracker:
    def __init__(self, max_history=50, max_age=10): # Added max_age
        self.track_history = defaultdict(lambda: deque(maxlen=max_history))
        self.next_id = 0
        self.active_tracks = {} # Stores {track_id: {'label': label, 'age': age}}
        self.max_age = max_age # Max frames a track can be missing before deletion

    def update_tracks(self, current_detections, iou_threshold=0.5):
        """Update object tracks with new detections"""
        if len(current_detections) != 3:
             print("Warning: current_detections tuple is not of length 3 (boxes, scores, labels). Skipping tracking.")
             return self.get_track_summary()

        boxes, scores, labels = current_detections
        
        if not isinstance(boxes, (list, np.ndarray)) or \
           not isinstance(scores, (list, np.ndarray)) or \
           not isinstance(labels, (list, np.ndarray)):
            print("Warning: Detections are not lists or numpy arrays. Skipping tracking.")
            return self.get_track_summary()
            
        if not (len(boxes) == len(scores) == len(labels)):
            print("Warning: Mismatch in length of boxes, scores, and labels. Skipping tracking.")
            return self.get_track_summary()

        matched_indices = set()
        matched_track_ids = set()

        # Increment age for all active tracks, mark initially as unmatched
        for track_id in list(self.active_tracks.keys()):
             self.active_tracks[track_id]['age'] += 1

        # Match detections to existing tracks
        if boxes: # Only try matching if there are current detections
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                best_track_id, best_iou = self.find_best_match(box, label, iou_threshold)

                if best_track_id is not None:
                    # Update existing track
                    self.track_history[best_track_id].append((box, score))
                    self.active_tracks[best_track_id]['age'] = 0 # Reset age on match
                    matched_indices.add(i)
                    matched_track_ids.add(best_track_id)

        # Create new tracks for unmatched detections
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if i not in matched_indices:
                track_id = self.next_id
                self.track_history[track_id].append((box, score))
                self.active_tracks[track_id] = {'label': label, 'age': 0}
                self.next_id += 1
                matched_track_ids.add(track_id) # Newly created tracks are also considered "active" this frame

        # Clean up tracks that are too old (weren't matched and exceeded max_age)
        self.cleanup_tracks() # Call the newly added function

        return self.get_track_summary()

    def find_best_match(self, box, label, iou_threshold):
        """Find best matching active track using IoU and label consistency"""
        best_iou = iou_threshold
        best_track_id = None

        for track_id, track_data in self.active_tracks.items():
            if track_data['label'] != label:
                continue
                
            # Get last known position if history exists
            if not self.track_history[track_id]:
                continue
            last_box = self.track_history[track_id][-1][0]
            iou = self.calculate_iou(box, last_box)
            
            if iou > best_iou:
                best_iou = iou
                best_track_id = track_id
        
        return best_track_id, best_iou # Return IoU as well

    # --- NEW FUNCTION ADDED ---
    def cleanup_tracks(self):
        """Remove tracks that haven't been seen for max_age frames."""
        lost_track_ids = []
        for track_id, track_data in self.active_tracks.items():
            if track_data['age'] > self.max_age:
                lost_track_ids.append(track_id)

        for track_id in lost_track_ids:
            print(f"Removing lost track ID: {track_id}")
            del self.active_tracks[track_id]
            # Optionally delete from history too, or keep it
            if track_id in self.track_history:
                 del self.track_history[track_id]
    # --- END OF NEW FUNCTION ---

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        # Ensure boxes have 4 elements
        if len(box1) != 4 or len(box2) != 4: return 0
        
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        box1_x2, box1_y2 = x1 + w1, y1 + h1
        box2_x2, box2_y2 = x2 + w2, y2 + h2
        
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        # Add small epsilon to avoid division by zero
        return inter_area / (union_area + 1e-6)

    def get_track_summary(self):
        """Get summary of currently active tracks"""
        summary = {}
        for track_id, track_data in self.active_tracks.items():
             if self.track_history[track_id]: # Check if history exists
                current_box, current_score = self.track_history[track_id][-1]
                summary[track_id] = {
                    'box': current_box,
                    'score': current_score,
                    'label': track_data['label'],
                    'age': track_data['age'], # Include age
                    'track_length': len(self.track_history[track_id])
                    # 'stability': self.calculate_track_stability(history) # Stability calc removed for simplicity
                }
        return summary