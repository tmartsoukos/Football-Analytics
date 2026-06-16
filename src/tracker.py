# =============================================================================
# tracker.py
# Μονάδα tracking για το Football Analytics pipeline.
# (Video Basics Lab: frame-by-frame επεξεργασία, persistent state)
#
# Το object detection είναι stateless — κοιτάει κάθε frame μόνο του.
# Το tracking λύνει αυτό: δίνει σταθερό ID σε κάθε αντικείμενο ανά frame.
#
# Αλγόριθμος: ByteTrack σε δύο βήματα:
#   1. Kalman Filter     — προβλέπει ΠΟΥ θα βρίσκεται το αντικείμενο
#                          στο επόμενο frame (βάσει ταχύτητας + θέσης).
#   2. Hungarian Algorithm — κάνει optimal matching νέων detections
#                            με existing tracks (minimum total distance).
#
# Αποτέλεσμα: "ο παίκτης #7 στο frame 42 είναι ο ίδιος με το frame 41"
# ακόμα και αν επικαλύπτονται ή κινούνται γρήγορα.
# =============================================================================


# Δεν κάνουμε import ByteTrack απευθείας — το Ultralytics το εσωκλείει
# και το εκθέτει μέσω model.track() με config "bytetrack.yaml".


class FootballTracker:
    """
    Wraps το YOLOv8 + ByteTrack pipeline για football footage.

    Δέχεται raw frame + YOLO model → επιστρέφει λίστα tracked objects
    με σταθερό ID, bounding box και class label ανά frame.

    Ο YOLO model δεν δημιουργείται εδώ — περνάει από έξω (main.py).
    Έτσι ο tracker είναι lightweight και model-agnostic.
    """

    # Οι μόνες κλάσεις COCO που μας ενδιαφέρουν:
    #   0  = person  (παίκτες + διαιτητής)
    #   32 = sports ball
    ALLOWED_CLASS_IDS = [0, 32]

    def track_frame(self, detector_model, frame) -> list:
        """
        Τρέχει ByteTrack σε ένα frame και επιστρέφει τα tracked objects.

        Παράμετροι:
            detector_model : φορτωμένο Ultralytics YOLO object (από main.py).
            frame          : BGR image ως NumPy array, shape (H, W, 3).

        Επιστρέφει:
            list of dicts:
                {
                  "bbox"     : [x1, y1, x2, y2],  # float
                  "track_id" : int,                 # persistent ID
                  "class_id" : int                  # 0 ή 32
                }
        """

        tracked_objects = []

        # --- Βήμα 1: YOLOv8 με ByteTrack ενεργό ---
        #
        # persist=True: κρατάει ζωντανή την εσωτερική κατάσταση του Kalman Filter
        # μεταξύ calls. Χωρίς αυτό κάθε frame θεωρείται πρώτο → reset IDs.
        # "bytetrack.yaml": επιλέγει ByteTrack αντί BoT-SORT (ταχύτερο,
        # καλύτερο σε low-confidence detections — σημαντικό για τη μπάλα).
        results = detector_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        # --- Βήμα 2: Guard αν δεν επέστρεψε τίποτα ---
        #
        # model.track() μπορεί να επιστρέψει κενή λίστα αν δεν βρήκε τίποτα.
        # Το index [0] σε κενή λίστα → IndexError → crash. Προλαμβάνουμε.
        if results is None or len(results) == 0:
            return tracked_objects

        # Ένα Results object ανά input image — εμείς στέλνουμε ένα frame.
        result = results[0]

        # Αν δεν υπάρχουν bounding boxes, επιστρέφουμε αδεια λίστα.
        if result.boxes is None or len(result.boxes) == 0:
            return tracked_objects

        # --- Βήμα 3: Loop σε κάθε detection ---
        for box in result.boxes:

            class_id = int(box.cls[0])

            # Φιλτράρουμε — κρατάμε μόνο person και sports ball
            if class_id not in self.ALLOWED_CLASS_IDS:
                continue

            # --- Βήμα 4: Εξαγωγή track ID από ByteTrack ---
            #
            # Η μπάλα (class 32) είναι μικρή και γρήγορη — ο Kalman Filter
            # είναι βελτιστοποιημένος για human-scale κίνηση, οπότε μερικές
            # φορές "χάνει" τη μπάλα (box.id = None).
            # Σε αυτή την περίπτωση βάζουμε track_id = -1 (sentinel value)
            # ώστε το main.py να ξέρει να ενεργοποιήσει το fallback detection.
            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1  # temporarily lost / untracked

            # xyxy format: πάνω-αριστερή + κάτω-δεξιά γωνία
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            tracked_object = {
                "bbox":     [x1, y1, x2, y2],
                "track_id": track_id,
                "class_id": class_id,
            }

            tracked_objects.append(tracked_object)

        return tracked_objects
