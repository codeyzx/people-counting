"""
Person tracking module for maintaining IDs across frames.

This module implements centroid-based tracking to assign and maintain unique
IDs for detected people across video frames.
"""

import logging
from typing import List, Dict
import numpy as np
from .models import Detection, TrackedPerson


logger = logging.getLogger(__name__)


class PersonTracker:
    """Tracks detected people across frames and assigns unique IDs.
    
    Uses centroid-based tracking to match detections across consecutive frames
    and maintains a real-time count of people currently in the frame.
    """
    
    def __init__(self, max_distance: float = 50.0):
        """Initialize the tracker.
        
        Args:
            max_distance: Maximum distance (in pixels) for matching detections
                         across frames. Detections farther apart are considered
                         different people.
        """
        self.max_distance = max_distance
        self.next_id = 0
        self.tracked_people: Dict[int, TrackedPerson] = {}
        
        logger.info(f"PersonTracker initialized with max_distance={max_distance}")
    
    def update(self, detections: List[Detection]) -> List[TrackedPerson]:
        """Update tracker with new detections.
        
        Matches new detections with existing tracked people based on centroid
        distance. Assigns new IDs to unmatched detections and updates the
        total count.
        
        Args:
            detections: List of detections from current frame
            
        Returns:
            List of TrackedPerson objects with assigned IDs
        """
        current_tracked = []
        
        if not detections:
            # No detections in this frame, clear tracked people
            self.tracked_people.clear()
            return current_tracked
        
        # If no existing tracks, assign new IDs to all detections
        if not self.tracked_people:
            for detection in detections:
                person_id = self._get_next_id()
                tracked_person = TrackedPerson(
                    person_id=person_id,
                    bbox=detection.bbox,
                    centroid=detection.centroid,
                    confidence=detection.confidence
                )
                self.tracked_people[person_id] = tracked_person
                current_tracked.append(tracked_person)
            
            logger.debug(f"Assigned new IDs to {len(detections)} detections")
            return current_tracked
        
        # Match existing tracks with new detections
        used_detection_indices = set()
        used_track_ids = set()
        
        # Calculate distances between all existing tracks and new detections
        for track_id, tracked_person in self.tracked_people.items():
            min_distance = float('inf')
            best_match_idx = -1
            
            for idx, detection in enumerate(detections):
                if idx in used_detection_indices:
                    continue
                
                # Calculate Euclidean distance between centroids
                distance = self._calculate_distance(
                    tracked_person.centroid,
                    detection.centroid
                )
                
                if distance < min_distance and distance < self.max_distance:
                    min_distance = distance
                    best_match_idx = idx
            
            # If a match was found, update the track
            if best_match_idx != -1:
                detection = detections[best_match_idx]
                tracked_person = TrackedPerson(
                    person_id=track_id,
                    bbox=detection.bbox,
                    centroid=detection.centroid,
                    confidence=detection.confidence
                )
                self.tracked_people[track_id] = tracked_person
                current_tracked.append(tracked_person)
                used_detection_indices.add(best_match_idx)
                used_track_ids.add(track_id)
        
        # Remove tracks that weren't matched (person left the frame)
        tracks_to_remove = [tid for tid in self.tracked_people.keys() if tid not in used_track_ids]
        for track_id in tracks_to_remove:
            del self.tracked_people[track_id]
        
        # Assign new IDs to unmatched detections (new people)
        for idx, detection in enumerate(detections):
            if idx not in used_detection_indices:
                person_id = self._get_next_id()
                tracked_person = TrackedPerson(
                    person_id=person_id,
                    bbox=detection.bbox,
                    centroid=detection.centroid,
                    confidence=detection.confidence
                )
                self.tracked_people[person_id] = tracked_person
                current_tracked.append(tracked_person)
        
        logger.debug(f"Tracking {len(current_tracked)} people")
        return current_tracked
    
    def get_current_count(self) -> int:
        """Get the current count of people in the frame.
        
        Returns:
            Number of people currently being tracked (real-time count)
        """
        return len(self.tracked_people)
    
    def _get_next_id(self) -> int:
        """Generate and return the next unique ID.
        
        Returns:
            Next unique person ID
        """
        person_id = self.next_id
        self.next_id += 1
        return person_id
    
    def _calculate_distance(self, point1: tuple, point2: tuple) -> float:
        """Calculate Euclidean distance between two points.
        
        Args:
            point1: First point as (x, y)
            point2: Second point as (x, y)
            
        Returns:
            Euclidean distance between the points
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
