# =============================================================================
# visualizer.py
# Visualization module για το Football Analytics pipeline.
# (Filters Lab: GaussianBlur, addWeighted | Image Basics Lab: masking, colormap)
#
# Δύο αρμοδιότητες:
#
#   1. REAL-TIME ANNOTATION — bounding boxes, track IDs, ball markers σε
#      κάθε frame για live inspection της ανίχνευσης και κατάταξης.
#
#   2. TACTICAL HEATMAP — συσσώρευση foot-positions ανά frame σε 2D density
#      map → smoothed, color-coded εικόνα που αποκαλύπτει zones activity.
#
# ΘΕΩΡΙΑ — 2D Accumulation Density Map:
#   Το pitch ως grid (ένα cell ανά pixel). Κάθε φορά που παίκτης στέκεται
#   εκεί → +1 στο cell. Μετά από χιλιάδες frames: raw "visit counts".
#   Αυτό είναι 2D histogram / discrete spatial probability distribution.
#
# ΓΙΑΤΙ GAUSSIAN SMOOTHING; (Filters Lab)
#   Raw accumulator = scattered bright dots σε μαύρο background.
#   Convolution με Gaussian kernel → κάθε dot γίνεται smooth "bump".
#   Αποτέλεσμα μοιάζει με PDF — πολύ πιο readable για tactical analysis.
#
# ROI MASK — SIDELINE NOISE:
#   YOLOv8 ανιχνεύει και coaches/substitutes/ball-boys στα sidelines.
#   Είναι quasi-stationary → δημιουργούν artificial hot-spots στις άκρες.
#   Λύση: polygon ROI που ορίζει το active pitch boundary. Detections
#   εκτός ROI αγνοούνται σε heatmap και bounding box drawing.
# =============================================================================

import cv2
import numpy as np
from collections import deque


class FootballVisualizer:
    """
    Handles όλο το visual output του Football Analytics pipeline:
      - Per-frame bounding box annotation με team colors και track IDs.
      - Incremental heatmap accumulation σε όλο το βίντεο.
      - Final heatmap generation (smoothed, normalized, color-mapped).
    """

    # Visual style constants — αλλαγή χρωμάτων/πάχους μόνο εδώ
    COLOR_TEAM_0  = (0,   0,   255)  # Red       (BGR) — Team 0
    COLOR_TEAM_1  = (255, 0,   0  )  # Blue      (BGR) — Team 1
    COLOR_UNKNOWN = (128, 128, 128)  # Grey      (BGR) — unclassified
    COLOR_BALL    = (0,   255, 255)  # Yellow    (BGR) — sports ball
    COLOR_TRAIL   = (0,   255, 128)  # Neon green(BGR) — ball trail

    # Trail history: 15 frames = 0.5 sec στα 30fps
    # Αρκεί για direction/curve, δεν clutters το frame μετά από sharp turn
    BALL_TRAIL_LENGTH = 15
    BOX_THICKNESS = 2
    FONT          = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE    = 0.6
    FONT_THICKNESS = 2

    def __init__(self, width: int, height: int):
        """
        Αρχικοποίηση visualizer για video των δοσμένων διαστάσεων.

        Παράμετροι:
            width  (int): Frame width σε pixels.
            height (int): Frame height σε pixels.

        Χρησιμοποιούμε float32 accumulators για GaussianBlur + normalize
        accuracy — πιο ακριβές από int.
        """
        self.width  = width
        self.height = height

        # Γιατί shape (height, width) και όχι (width, height);
        # NumPy + OpenCV indexing: array[y, x] → shape = (H, W).

        # --- Δύο ξεχωριστοί accumulators, ένας ανά ομάδα ---
        #
        # Γιατί split ανά ομάδα;
        #   1. DEFENSIVE BLOCK ANALYSIS — Team 1 heatmap αποκαλύπτει defensive line.
        #   2. OFFENSIVE ZONES — Team 0 heatmap δείχνει half-spaces των forwards.
        #   3. TERRITORIAL DOMINANCE — Side-by-side σύγκριση: ποια ομάδα
        #      κυριάρχησε σε ποιο zone.
        #   Ένας shared accumulator θα έκρυβε αυτή τη δομή.
        self.accumulator_team0 = np.zeros((height, width), dtype=np.float32)
        self.accumulator_team1 = np.zeros((height, width), dtype=np.float32)

        # --- Pitch ROI polygon ---
        #
        # Γιατί polygon και όχι rectangle;
        #   Λόγω perspective projection, το pitch φαίνεται ως trapezoid.
        #   6-point polygon προσεγγίζει καλύτερα αυτό το σχήμα από ορθογώνιο.
        #
        # Βαθμονόμηση: άνοιξε representative frame, hover πάνω στις pitch
        # corners και διάβασε (x,y) pixel coordinates.
        #
        # bottom_roi_y = height - 15:
        #   Αφήνει 15px buffer για να εξαιρέσει bench/scoreboard pixels.
        #   Adaptive formula → δουλεύει για 720p (→705) και 1080p (→1065).
        bottom_roi_y = height - 15

        self.pitch_roi_polygon = np.array([
            [50,          200         ],   # top-left     (far touchline, left)
            [width - 50,  200         ],   # top-right    (far touchline, right)
            [width - 50,  bottom_roi_y],   # bottom-right (near touchline, right)
            [50,          bottom_roi_y],   # bottom-left  (near touchline, left)
        ], dtype=np.int32)

        # Pre-computed binary ROI mask για fast point-in-polygon lookups.
        # roi_mask[y, x] == 255 → inside pitch | == 0 → outside
        # Ένα array read κάθε frame, όχι geometry function call.
        self.roi_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(self.roi_mask, [self.pitch_roi_polygon], color=255)

        # --- Ball trajectory deque ---
        #
        # deque με fixed maxlen = ideal sliding-window structure:
        #   - append O(1)
        #   - overflow: αυτόματα αφαιρείται η παλαιότερη θέση
        #   - maxlen enforces window size at data-structure level
        # Κάθε element: tuple (cx, cy) = ball center pixel.
        self.ball_history = deque(maxlen=self.BALL_TRAIL_LENGTH)

    # -------------------------------------------------------------------------
    # METHOD 1 — REAL-TIME ANNOTATION
    # -------------------------------------------------------------------------

    def draw_predictions(self, frame: np.ndarray, tracked_objects: list,
                         classifier=None) -> np.ndarray:
        """
        Σχεδιάζει bounding boxes, team labels, track IDs και ball markers
        σε αντίγραφο του frame.

        Design: τα team labels υπολογίζονται στο main.py και αποθηκεύονται
        στο obj["team"]. Εδώ απλώς τα διαβάζουμε — ο visualizer δεν κάνει
        classify. Single Responsibility Principle.

        Παράμετροι:
            frame           (np.ndarray): BGR video frame (δεν τροποποιείται).
            tracked_objects (list)      : Tracked objects με 'team' key.
            classifier      : Unused — kept for API compatibility.

        Επιστρέφει:
            np.ndarray: Annotated BGR frame.
        """
        # Copy για να μην τροποποιούμε το original frame (χρειάζεται για heatmap)
        annotated_frame = frame.copy()

        for obj in tracked_objects:
            bbox      = obj["bbox"]
            track_id  = obj["track_id"]
            class_id  = obj["class_id"]

            # Float bbox coords → integers για OpenCV draw functions
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # --- ROI check για ΠΑΙΚΤΕΣ ---
            #
            # Ελέγχουμε αν το foot (κάτω-κεντρικό bbox) είναι μέσα στο pitch ROI.
            # Coaches/substitutes στον bench → no bounding box → cleaner annotation.
            #
            # Γιατί foot και όχι center; Το foot είναι όπου ο παίκτης
            # αγγίζει το έδαφος — πιο accurate για edge cases.
            # Η μπάλα (class 32) εξαιρείται — corner kicks την βγάζουν εκτός.
            if class_id == 0:
                foot_x = max(0, min((x1 + x2) // 2, self.width  - 1))
                foot_y = max(0, min(y2,              self.height - 1))
                if self.roi_mask[foot_y, foot_x] == 0:
                    continue  # εκτός pitch ROI — skip

            # ------------------------------------------------------------------
            # Case A: ΠΑΙΚΤΗΣ (class 0 = person)
            # ------------------------------------------------------------------
            if class_id == 0:

                # Team label: 0 = Team 0, 1 = Team 1, -1 = unknown
                team = obj.get("team", -1)

                # Χρώμα bbox βάσει ομάδας
                if team == 0:
                    box_color = self.COLOR_TEAM_0
                elif team == 1:
                    box_color = self.COLOR_TEAM_1
                else:
                    box_color = self.COLOR_UNKNOWN

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2),
                              box_color, self.BOX_THICKNESS)

                # Label: "T0 #7" = Team 0, Track ID 7
                team_str = f"T{team}" if team >= 0 else "T?"
                id_str   = f"#{track_id}" if track_id != -1 else "#?"
                label    = f"{team_str} {id_str}"

                # Ελαφρά πάνω από το bbox — clamp ώστε να μένει εντός frame
                label_y = max(y1 - 8, 15)
                cv2.putText(annotated_frame, label, (x1, label_y),
                            self.FONT, self.FONT_SCALE, box_color, self.FONT_THICKNESS)

            # ------------------------------------------------------------------
            # Case B: ΜΠΑΛΑ (class 32 = sports ball)
            # ------------------------------------------------------------------
            elif class_id == 32:

                # Κέντρο bbox — για τη μπάλα κύκλος είναι πιο natural από ορθογώνιο
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Αποθήκευση στο trajectory history (deque auto-manages overflow)
                # Το trail είναι purely visual → ανήκει στον visualizer, όχι στο main.
                self.ball_history.append((center_x, center_y))

                # --- Trail drawing ---
                #
                # Consecutive (prev, curr) pairs → line segments → polyline trail.
                # Fade effect: παλαιότερες θέσεις = λεπτότερη γραμμή.
                # Formula: i ∈ [1, N-1] → thickness ∈ [1, 4] linearly.
                history_list = list(self.ball_history)
                num_points   = len(history_list)

                for i in range(1, num_points):
                    pt_prev = history_list[i - 1]
                    pt_curr = history_list[i]

                    thickness = max(1, int(4 * i / max(num_points - 1, 1)))

                    cv2.line(annotated_frame, pt_prev, pt_curr,
                             self.COLOR_TRAIL, thickness)

                # Filled yellow circle + dark outline για καλύτερη ορατότητα
                radius = max(((x2 - x1) + (y2 - y1)) // 4, 5)
                cv2.circle(annotated_frame, (center_x, center_y),
                           radius, self.COLOR_BALL, thickness=-1)
                cv2.circle(annotated_frame, (center_x, center_y),
                           radius, (0, 0, 0), thickness=1)

                cv2.putText(annotated_frame, f"Ball #{track_id}",
                            (x1, max(y1 - 8, 15)),
                            self.FONT, self.FONT_SCALE, self.COLOR_BALL, self.FONT_THICKNESS)

        return annotated_frame

    # -------------------------------------------------------------------------
    # METHOD 2 — POSSESSION OVERLAY
    # -------------------------------------------------------------------------

    def draw_possession_overlay(self, frame: np.ndarray,
                                frames_t0: int, frames_t1: int) -> np.ndarray:
        """
        Σχεδιάζει semi-transparent possession statistics bar στο frame.
        (Filters Lab: cv2.addWeighted για blending)

        Possession Metric:
            Possession_T0 (%) = frames_T0 / (frames_T0 + frames_T1) × 100
            όπου frames_T0 = frames που παίκτης Team 0 ήταν πιο κοντά στη μπάλα
            (Euclidean distance < threshold). Στατιστικό proxy — δεν ξεχωρίζει
            controlled touch από 50-50 ball, αλλά correlates καλά με
            professional possession metrics σε αρκετά frames.

        Γιατί Euclidean distance;
            d = sqrt((x_ball - x_player)² + (y_ball - y_player)²)
            Threshold 80px → "possession zone". Αν παίκτης είναι εκτός →
            δεν αποδίδεται κατοχή (αποφεύγουμε false attributions mid-air).

        UI: Dark panel με cv2.addWeighted (α=0.6) → 60% opaque, αλλά
            τα pitch markings φαίνονται ακόμα — Filters Lab technique.

        Παράμετροι:
            frame     (np.ndarray): Annotated BGR frame.
            frames_t0 (int)       : Frames που Team 0 είχε κατοχή.
            frames_t1 (int)       : Frames που Team 1 είχε κατοχή.

        Επιστρέφει:
            np.ndarray: Frame με possession overlay.
        """
        total = frames_t0 + frames_t1

        # Default 50/50 αν δεν έχει αποδοθεί κατοχή ακόμα
        if total == 0:
            pct_t0, pct_t1 = 50.0, 50.0
        else:
            pct_t0 = (frames_t0 / total) * 100.0
            pct_t1 = (frames_t1 / total) * 100.0

        label = (f"POSSESSION  |  "
                 f"T0 (White): {pct_t0:.1f}%    "
                 f"T1 (Dark): {pct_t1:.1f}%")

        # --- Semi-transparent dark background panel (Filters Lab: addWeighted) ---
        #
        # Plain filled rectangle → μπλοκάρει το βίντεο από κάτω.
        # addWeighted: output = α * overlay + (1-α) * original
        # α=0.6 → 60% panel, 40% original → readable αλλά pitch φαίνεται.
        panel_h  = 40
        panel_y1 = self.height - panel_h
        panel_y2 = self.height

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_y1), (self.width, panel_y2),
                      (20, 20, 20), thickness=-1)       # near-black panel
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # --- Κεντραρισμένο text πάνω στο panel ---
        text_size, _ = cv2.getTextSize(label, self.FONT, 0.65, 2)
        text_x = (self.width - text_size[0]) // 2
        text_y = panel_y1 + (panel_h + text_size[1]) // 2

        cv2.putText(frame, label, (text_x, text_y),
                    self.FONT, 0.65, (255, 255, 255), 2)  # white text

        return frame

    # -------------------------------------------------------------------------
    # METHOD 3 — HEATMAP ACCUMULATION
    # -------------------------------------------------------------------------

    def update_heatmap(self, tracked_objects: list):
        """
        Αυξάνει τον accumulator της σωστής ομάδας στη foot position
        κάθε tracked παίκτη — Team 0 → accumulator_team0, Team 1 → accumulator_team1.

        Γιατί FOOT και όχι center;
            Το center είναι στη μέση — δεν ακουμπάει το έδαφος.
            Το y2 (κάτω bbox) → foot position → accurate spatial projection.

        Παράμετροι:
            tracked_objects (list): Tracked objects με 'team' key από main.py.
        """
        for obj in tracked_objects:

            # Μόνο παίκτες (class 0) — η μπάλα δεν πηγαίνει στο heatmap
            if obj["class_id"] != 0:
                continue

            team = obj.get("team", -1)

            # Unclassified detections (team == -1) → skip
            # Αν συμπεριληφθούν, θα μολύνουν έναν από τους accumulators
            if team not in (0, 1):
                continue

            bbox = obj["bbox"]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            # Foot position: bottom-center του bbox
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)

            # Clamp σε valid frame coordinates
            foot_x = max(0, min(foot_x, self.width  - 1))
            foot_y = max(0, min(foot_y, self.height - 1))

            # ROI check: coaches/substitutes → artificial hot-spots στις άκρες → skip
            if self.roi_mask[foot_y, foot_x] == 0:
                continue

            # Routing στον σωστό accumulator — το κλειδί για team-separated analysis
            if team == 0:
                self.accumulator_team0[foot_y, foot_x] += 1.0
            else:
                self.accumulator_team1[foot_y, foot_x] += 1.0

    # -------------------------------------------------------------------------
    # METHOD 4 — TEAM-SPECIFIC HEATMAP GENERATION
    # -------------------------------------------------------------------------

    def generate_team_heatmap(self, team_id: int) -> np.ndarray:
        """
        Μετατρέπει τον raw accumulator μιας ομάδας σε publication-quality heatmap.

        TACTICAL INTERPRETATION:
            Team 0 heatmap: hot zones αντίπαλης μισής → territorial control.
            Team 1 heatmap: deep hot zone → defensive low block ή mid press.
            Side-by-side: ποια ομάδα κυριάρχησε, ποια εκχώρησε flanks.

        Pipeline (ίδιο και για τις δύο ομάδες):
            1. Gaussian Blur  → discrete foot positions → continuous PDF
            2. Normalization  → rescale [min, max] → [0, 255]
            3. COLORMAP_JET   → blue (cold) → red (hot), standard analytics scheme

        Παράμετροι:
            team_id (int): 0 ή 1.

        Επιστρέφει:
            np.ndarray: BGR color image (H, W, 3) έτοιμο για cv2.imwrite().
        """
        # Επιλογή accumulator βάσει team_id
        if team_id == 0:
            accumulator = self.accumulator_team0
        else:
            accumulator = self.accumulator_team1

        # --- Βήμα 1: Gaussian Blur  (Filters Lab) ---
        #
        # Raw accumulator: sparse dots → convolve με Gaussian → smooth bumps.
        # blurred(x,y) = Σ accumulator(x',y') * G(x-x', y-y')
        # G: 2D Gaussian function με σ ≈ 22px (auto-derived από kernel size).
        #
        # kernel_size=151: σε 1080p ένας παίκτης = ~80-150px ύψος.
        # 151px kernel → γειτονικές foot positions blendάρονται ομαλά.
        # Πάντα ODD — OpenCV requirement για GaussianBlur.
        # sigmaX=0 → auto sigma: 0.3*((151-1)*0.5-1)+0.8 ≈ 22.3px
        kernel_size = 151
        blurred = cv2.GaussianBlur(accumulator, (kernel_size, kernel_size), sigmaX=0)

        # --- Βήμα 2: Normalization [0, 255] ---
        #
        # NORM_MINMAX: linearly maps [min_val, max_val] → [0, 255].
        # Κάθε ομάδα normalize ανεξάρτητα → relative activity zones per team,
        # όχι absolute comparison μεταξύ ομάδων.
        normalized = cv2.normalize(blurred, None, alpha=0, beta=255,
                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # --- Βήμα 3: COLORMAP_JET ---
        #
        # JET: 0 → Deep Blue (cold) | 128 → Green/Cyan | 255 → Deep Red (hot)
        # Standard sports analytics color scheme (Wyscout, InStat, StatsBomb).
        # Coaches το διαβάζουν αμέσως χωρίς legend explanation.
        heatmap_color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        return heatmap_color
