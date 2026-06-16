# =============================================================================
# detector.py
# Μονάδα ανίχνευσης αντικειμένων με χρήση του μοντέλου YOLOv8.
# Χρησιμοποιείται για την ανίχνευση παικτών και της μπάλας σε βίντεο
# ποδοσφαίρου. Αποτελεί μέρος πτυχιακής εργασίας για Football Analytics.
# =============================================================================

import numpy as np
from ultralytics import YOLO


class FootballDetector:
    """
    Κλάση που αναλαμβάνει την ανίχνευση αντικειμένων σε καρέ βίντεο
    ποδοσφαίρου, χρησιμοποιώντας το μοντέλο YOLOv8.

    Φιλτράρει τις ανιχνεύσεις ώστε να κρατάει μόνο:
      - Κλάση 0 : 'person'      (παίκτες και διαιτητές)
      - Κλάση 32: 'sports ball' (η μπάλα ποδοσφαίρου)
    """

    # Ορισμός των κλάσεων COCO που μας ενδιαφέρουν.
    # Το YOLOv8 εκπαιδεύτηκε στο dataset COCO που έχει 80 κλάσεις.
    # Εμείς χρειαζόμαστε μόνο τις παρακάτω δύο για το project μας.
    ALLOWED_CLASS_IDS = [0, 32]  # 0=person, 32=sports ball

    def __init__(self, model_path: str = "yolov8m.pt"):
        """
        Αρχικοποίηση του ανιχνευτή και φόρτωση του μοντέλου YOLOv8.

        Παράμετροι:
            model_path (str): Το μονοπάτι προς τα βάρη του μοντέλου (.pt αρχείο).
                              Αν δεν δοθεί, χρησιμοποιείται το 'yolov8m.pt'
                              (medium μέγεθος — καλή ισορροπία ταχύτητας/ακρίβειας).
        """
        # Αποθηκεύουμε το μονοπάτι για μετέπειτα χρήση αν χρειαστεί
        self.model_path = model_path

        # Φόρτωση μοντέλου — αν δεν υπάρχει τοπικά, κατεβαίνει αυτόματα
        print(f"[FootballDetector] Φόρτωση μοντέλου από: {self.model_path}")
        self.model = YOLO(self.model_path)
        print("[FootballDetector] Το μοντέλο φορτώθηκε επιτυχώς.")

    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.3) -> list:
        """
        Εκτελεί ανίχνευση αντικειμένων σε ένα μεμονωμένο καρέ (frame) εικόνας.

        Παράμετροι:
            frame (np.ndarray): BGR frame, σχήμα (H, W, 3).
            confidence_threshold (float): Ελάχιστο confidence score. Προεπιλογή: 0.3.

        Επιστρέφει:
            list: Λίστα dicts με κλειδιά 'bbox', 'confidence', 'class_id'.
        """
        # Λίστα αποτελεσμάτων — γεμίζει μόνο με όσα περνούν τα φίλτρα
        filtered_detections = []

        # verbose=False → δεν τυπώνει αποτελέσματα σε κάθε frame
        results = self.model(frame, verbose=False)

        # Ένα result ανά εικόνα — εδώ στέλνουμε μία κάθε φορά
        result = results[0]

        # Αδεια λίστα αν δεν βρέθηκαν detections
        if result.boxes is None:
            return filtered_detections

        # Loop σε κάθε bounding box
        for box in result.boxes:

            # Εξαγωγή class ID
            class_id = int(box.cls[0])

            # Κρατάμε μόνο person (0) και sports ball (32)
            if class_id not in self.ALLOWED_CLASS_IDS:
                continue

            confidence = float(box.conf[0])

            # Φίλτρο εμπιστοσύνης — αποφεύγουμε false positives
            if confidence < confidence_threshold:
                continue

            # xyxy format: πάνω-αριστερή και κάτω-δεξιά γωνία
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detection = {
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "class_id": class_id,
            }

            filtered_detections.append(detection)

        return filtered_detections

    def detect_ball(self, frame: np.ndarray,
                    confidence_threshold: float = 0.10) -> dict | None:
        """
        Dedicated pass ανίχνευσης μπάλας με χαμηλότερο confidence threshold.

        Γιατί ξεχωριστή μέθοδος;
            Η μπάλα στο broadcast είναι μικρή (~20-30px), ταχύκινητη και
            συχνά motion-blurred. Το YOLOv8m (COCO pre-trained) δεν ξέρει
            broadcast football — τα confidence scores σπάνια ξεπερνούν 0.15.
            Το standard threshold 0.30 για παίκτες σκοτώνει σχεδόν κάθε
            ball detection (diagnostics: 2/57 frames, conf~0.11).

            Λύση: class-specific threshold:
              - Παίκτες (class 0) : conf >= 0.30 — υψηλό για να αποφύγουμε
                                    false positives σε θεατές και διαφημίσεις.
              - Μπάλα (class 32)  : conf >= 0.10 — χαμηλό γιατί υπάρχει
                                    μία μόνο μπάλα, οπότε το FP rate είναι μικρό.

        Γνωστός περιορισμός (domain gap):
            Ακόμα και με conf=0.10, το recall παραμένει ~1-5% των frames
            γιατί το COCO δεν έχει broadcast football footage, τα anchor
            boxes δεν είναι φτιαγμένα για τόσο μικρά αντικείμενα, και το
            motion blur σε 30fps αλλοιώνει τα χαρακτηριστικά. Για production
            χρειάζεται fine-tuned μοντέλο (SoccerNet, Roboflow Football).

        Παράμετροι:
            frame                (np.ndarray): BGR video frame.
            confidence_threshold (float)     : Ελάχιστο confidence για τη μπάλα.

        Επιστρέφει:
            dict | None: Detection dict ή None αν δεν βρέθηκε μπάλα.
        """
        # Τρέχουμε predict με χαμηλό conf ώστε να μην κόβει εκεί το Ultralytics
        results = self.model.predict(frame, verbose=False, conf=confidence_threshold)

        if not results or results[0].boxes is None:
            return None

        best_ball = None
        best_conf = 0.0

        for box in results[0].boxes:
            if int(box.cls[0]) != 32:
                continue  # μας ενδιαφέρει μόνο η κλάση 32 (sports ball)

            conf = float(box.conf[0])
            if conf < confidence_threshold:
                continue

            # Κρατάμε το detection με το υψηλότερο confidence
            # Σε διφορούμενα frames το YOLOv8 μπορεί να δώσει δύο κοντινά detections
            if conf > best_conf:
                best_conf = conf
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                best_ball = {
                    "bbox":       [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id":   32,
                    "track_id":   -1,   # δεν έχει ByteTrack ID από raw predict
                    "team":       None,
                }

        return best_ball
