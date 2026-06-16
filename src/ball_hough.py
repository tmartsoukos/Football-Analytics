# =============================================================================
# ball_hough.py
# Κλασική CV ανίχνευση μπάλας με Hough Circle Transform.
# Χρησιμοποιείται ως Tier 3 fallback στο 3-tier cascade detection chain.
#
# ΓΙΑΤΙ ΥΠΑΡΧΕΙ ΑΥΤΟ ΤΟ MODULE:
#   Το YOLOv8m (COCO pre-trained) έχει recall ~1-5% για τη μπάλα στο
#   broadcast footage — πολύ μικρός στόχος, motion blur, domain gap.
#   Αυτό το module δεν χρειάζεται καθόλου μοντέλο — εφαρμόζει κλασική CV
#   όπως ακριβώς διδαχθήκαμε στα εργαστήρια:
#
#     Filters Lab      → cv2.GaussianBlur πριν edge detection
#     Edge Detection Lab → cv2.HoughCircles (εσωτερικά τρέχει Canny)
#     Image Basics Lab → grayscale conversion, pixel-level masking
#
# ΠΩΣ ΛΕΙΤΟΥΡΓΕΙ ΤΟ HOUGH CIRCLE TRANSFORM:
#   Βήμα 1 — Canny Edge Detection (εσωτερικά στο HoughCircles):
#     Κάθε pixel στην ακμή κύκλου παράγει gradient που δείχνει προς το κέντρο.
#     Κάθε pixel "ψηφίζει" για όλα τα πιθανά κέντρα στην κατεύθυνση του gradient.
#
#   Βήμα 2 — Accumulator Voting:
#     2D accumulator array μετράει ψήφους ανά υποψήφιο κέντρο (x, y).
#     Τα cells με πολλές ψήφους αντιστοιχούν σε πραγματικά κέντρα κύκλων.
#
#   Βήμα 3 — Peak Detection:
#     Τοπικά maxima στον accumulator (πάνω από param2) → detected circles.
#
#   Γιατί δουλεύει για μπάλα:
#     - Κυκλική σιλουέτα πάνω στο πράσινο γκαζόν → ισχυρή contrast ακμή.
#     - Λευκή μπάλα → mean pixel intensity >> δέρμα/μαλλιά (brightness filter).
#     - Μία μόνο μπάλα → εύκολο false positive suppression.
# =============================================================================

import cv2
import numpy as np


class BallHoughDetector:
    """
    Ανιχνεύει τη μπάλα σε broadcast footage με Hough Circle Transform.

    Pipeline ανά frame:
        BGR frame
          │
          ├─ 1. Grayscale conversion  (cv2.cvtColor)
          ├─ 2. Gaussian Blur         (cv2.GaussianBlur)  ← Filters Lab
          ├─ 3. Hough Circle Transform(cv2.HoughCircles)   ← Edge Detection Lab
          ├─ 4. ROI filter            (roi_mask lookup)
          ├─ 5. Brightness filter     (reject dark circles = κεφάλια)
          └─ 6. Best candidate select (πιο κοντά στο expected radius)

    Tier 3 fallback στο detection chain:
        Tier 1 — ByteTrack       (καλύτερο: persistent track_id)
        Tier 2 — YOLO detect_ball (conf=0.10)
        Tier 3 — ΑΥΤΗ Η ΚΛΑΣΗ   (κλασική CV, χωρίς μοντέλο)
    """

    # -------------------------------------------------------------------------
    # TUNING CONSTANTS — βαθμονομημένα για 1920×1080 broadcast footage.
    # Για άλλη ανάλυση (π.χ. 1280×720) scale ανάλογα.
    # -------------------------------------------------------------------------

    # Εύρος ακτίνας μπάλας σε pixels στο broadcast
    MIN_BALL_RADIUS = 2
    MAX_BALL_RADIUS = 5

    # Πιο συχνή ακτίνα στο broadcast — χρησιμοποιείται για scoring
    EXPECTED_BALL_RADIUS = 3

    # Ελάχιστη απόσταση μεταξύ δύο detected κύκλων — μία μπάλα στο γήπεδο
    MIN_CIRCLE_DIST = 200

    # param1 του HoughCircles: HIGH threshold για το εσωτερικό Canny.
    # Το LOW threshold = param1/2 = 25 αυτόματα.
    CANNY_HIGH_THRESHOLD = 40

    # Ελάχιστος αριθμός ψήφων για να αναφερθεί κύκλος.
    # Χαμηλότερο = περισσότεροι κύκλοι αλλά και περισσότερα false positives.
    HOUGH_ACCUMULATOR_THRESHOLD = 13

    # Ελάχιστη mean grayscale intensity μέσα στον κύκλο.
    # Λευκή μπάλα: ~180-240 | Κεφάλια (δέρμα/μαλλιά): ~90-150
    MIN_BALL_BRIGHTNESS = 135

    def detect(self, frame: np.ndarray, roi_mask: np.ndarray = None) -> dict | None:
        """
        Ανιχνεύει τη μπάλα σε ένα BGR frame με Hough Circle Transform.

        Παράμετροι:
            frame    (np.ndarray)       : BGR video frame, shape (H, W, 3).
            roi_mask (np.ndarray | None): Binary pitch mask — 255 = inside ROI,
                                          0 = sideline. None = skip ROI filter.

        Επιστρέφει:
            dict | None : Detection dict συμβατό με το pipeline:
                {
                  "bbox"       : [x1, y1, x2, y2],
                  "confidence" : 0.0,   # το Hough δεν δίνει score
                  "class_id"   : 32,    # sports ball
                  "track_id"   : -1,    # χωρίς ByteTrack ID
                  "team"       : None
                }
            None αν δεν βρέθηκε αξιόπιστος υποψήφιος.
        """

        # =============================================================
        # ΒΗΜΑ 1 — BGR → Grayscale  (Image Basics Lab)
        # =============================================================
        #
        # Το HoughCircles δουλεύει σε single-channel intensity image.
        # Gradient direction (για voting) υπολογίζεται από intensity —
        # δεν χρειάζεται χρώμα, θα πρόσθετε μόνο noise.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # =============================================================
        # ΒΗΜΑ 2 — Gaussian Blur  (Filters Lab)
        # =============================================================
        #
        # Το HoughCircles καλεί εσωτερικά Canny για ακμές. Grass texture,
        # jersey patterns και JPEG artifacts παράγουν μικρές spurious ακμές
        # που γεμίζουν τον accumulator με ψεύτικους κύκλους.
        #
        # Το Gaussian Blur εξομαλύνει pixel-level noise αλλά κρατάει
        # την ισχυρή κυκλική ακμή της μπάλας — ακριβώς η αντιστάθμιση
        # που είδαμε στο Filters Lab (Box vs Gaussian vs Median).
        #
        # kernel (9,9), sigmaX=2:
        #   9×9 αρκεί για να σβήσει grass texture (φύλλο ~2-4px) αλλά
        #   δεν καταστρέφει την ακμή της μπάλας (radius ≥ 6px → 38px περίμετρος).
        blurred = cv2.GaussianBlur(gray, (9, 9), sigmaX=2)

        # =============================================================
        # ΒΗΜΑ 3 — Hough Circle Transform  (Edge Detection Lab)
        # =============================================================
        #
        # cv2.HOUGH_GRADIENT: τρέχει Canny εσωτερικά, μετά voting.
        #
        # dp=1: accumulator ανάλυση = εικόνα ανάλυση (max precision).
        #       dp=2 → μισή ανάλυση, ταχύτερο αλλά λιγότερο ακριβές.
        #       Για μικρή μπάλα κάθε pixel έχει σημασία → dp=1.
        #
        # minDist=MIN_CIRCLE_DIST: merge κύκλων που είναι πολύ κοντά.
        #   Μία μπάλα → δεν θέλουμε duplicate detections για την ίδια.
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.MIN_CIRCLE_DIST,
            param1=self.CANNY_HIGH_THRESHOLD,
            param2=self.HOUGH_ACCUMULATOR_THRESHOLD,
            minRadius=self.MIN_BALL_RADIUS,
            maxRadius=self.MAX_BALL_RADIUS
        )

        # HoughCircles επιστρέφει None αν δεν βρήκε τίποτα
        if circles is None:
            return None

        # Output shape (1, N, 3): N κύκλοι, κάθε ένας (x_center, y_center, radius).
        # Squeeze + round → integers.
        circles = np.round(circles[0, :]).astype(int)

        # =============================================================
        # ΒΗΜΑ 4 — Φιλτράρισμα και επιλογή καλύτερου υποψηφίου
        # =============================================================

        best_detection = None
        best_score     = float("inf")

        for (cx, cy, r) in circles:

            # --- ROI filter ---
            #
            # Αποκλείουμε κύκλους εκτός pitch (logos διαφημίσεων, corner flags).
            if roi_mask is not None:
                cy_clamped = max(0, min(cy, roi_mask.shape[0] - 1))
                cx_clamped = max(0, min(cx, roi_mask.shape[1] - 1))
                if roi_mask[cy_clamped, cx_clamped] == 0:
                    continue  # εκτός pitch ROI — skip

            # --- Brightness filter ---
            #
            # Λευκή μπάλα: mean intensity ~180-240.
            # Κεφάλια παικτών (δέρμα/μαλλιά): ~90-150.
            # Αυτό το φίλτρο κόβει τα περισσότερα κεφάλια χωρίς να χάσουμε
            # μπάλες με ελαφρά σκίαση — ίδια τεχνική με το jersey masking
            # στο classifier.py (cv2.mean με mask).
            circle_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(circle_mask, (cx, cy), r, 255, thickness=-1)
            mean_brightness = cv2.mean(gray, mask=circle_mask)[0]

            if mean_brightness < self.MIN_BALL_BRIGHTNESS:
                continue  # πολύ σκοτεινό — μάλλον κεφάλι, όχι μπάλα

            # --- Scoring: επιλέγουμε τον κύκλο με ακτίνα πιο κοντά στην expected ---
            score = abs(r - self.EXPECTED_BALL_RADIUS)
            if score < best_score:
                best_score     = score
                best_detection = (cx, cy, r)

        if best_detection is None:
            return None

        # =============================================================
        # ΒΗΜΑ 5 — Μετατροπή σε pipeline-compatible bbox dict
        # =============================================================
        #
        # Το υπόλοιπο pipeline (possession, visualizer, heatmap) περιμένει
        # {bbox, confidence, class_id, track_id, team} format.
        cx, cy, r = best_detection

        # Smallest enclosing rectangle για τον κύκλο
        x1 = float(cx - r)
        y1 = float(cy - r)
        x2 = float(cx + r)
        y2 = float(cy + r)

        return {
            "bbox":       [x1, y1, x2, y2],
            "confidence": 0.0,   # το HoughCircles δεν δίνει confidence score
            "class_id":   32,    # COCO class 32 = sports ball
            "track_id":   -1,    # χωρίς ByteTrack ID
            "team":       None,  # η μπάλα δεν ανήκει σε ομάδα
        }
