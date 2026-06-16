# =============================================================================
# classifier.py
# Κατάταξη παικτών σε ομάδες με K-Means Clustering.
# (Image Basics Lab: HSV color space, masking, mean pixel values)
#
# ΠΡΟΒΛΗΜΑ: Πώς ξεχωρίζουμε τις δύο ομάδες αυτόματα, χωρίς labeled data;
#
# ΛΥΣΗ: Unsupervised Machine Learning → K-Means (k=2).
#
# ΠΡΟΣΕΓΓΙΣΗ (3 βήματα):
#   1. FEATURE EXTRACTION — torso crop, grass mask, [S, V] feature vector.
#   2. FITTING            — K-Means με k=2 σε clean, outlier-filtered samples.
#   3. PREDICTION         — assign νέο παίκτη στο κοντινότερο cluster center.
#
# ΓΙΑΤΙ [S, V] ΚΑΙ ΟΧΙ [H, S, V]:
#   Paris FC (λευκό) vs PSG (σκούρο navy/μαύρο) — και οι δύο near-achromatic.
#   Για λευκή φανέλα: H = undefined/noisy όταν S≈0.
#   Ο άξονας που τις ξεχωρίζει είναι η ΦΩΤΕΙΝΟΤΗΤΑ (Value):
#     Λευκό : S~0-30,  V~180-255  (πολύ φωτεινό)
#     Σκούρο: S~0-60,  V~20-100   (πολύ σκοτεινό)
#   Αφαιρούμε το H → μειώνουμε noise και δίνουμε στο K-Means μόνο
#   τα channels που φέρουν πραγματική πληροφορία για αυτές τις φανέλες.
# =============================================================================

import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    """
    Κατατάσσει παίκτες ποδοσφαίρου σε δύο ομάδες με K-Means clustering
    πάνω σε brightness features (Saturation + Value) εξαγόμενες από το
    torso του παίκτη μετά από masking του γκαζόν.

    Σχεδιασμένο για achromatic kit combinations όπως Λευκό vs Σκούρο.
    """

    # -------------------------------------------------------------------------
    # GRASS MASK CONSTANTS (HSV space)
    #
    # Pixels εντός [GRASS_LOWER, GRASS_UPPER] = γρασίδι → αφαιρούνται.
    #
    # Γιατί HSV για το mask; (Image Basics Lab)
    #   Το πράσινο γρασίδι έχει compact, well-defined Hue band στο HSV.
    #   Στο BGR χρειάζονται τρεις ταυτόχρονες ανισότητες που αλλάζουν
    #   με τις συνθήκες φωτισμού — το HSV είναι πολύ πιο σταθερό.
    #
    #   H: 35-75  → yellow-green έως pure green (με shadow margin)
    #   S: 40+    → δεν maskάρουμε λευκές/γκρι φανέλες (S~0)
    #   V: 40+    → δεν maskάρουμε πολύ σκοτεινά pixels
    GRASS_LOWER = np.array([35,  40,  40], dtype=np.uint8)
    GRASS_UPPER = np.array([75, 255, 255], dtype=np.uint8)

    # -------------------------------------------------------------------------
    # OUTLIER FILTER CONSTANTS
    #
    # MIN_JERSEY_PIXELS:
    #   Αν λιγότερα από τόσα non-grass pixels επιβιώσουν, το crop είναι
    #   too small/occluded → αναξιόπιστο feature → παραλείπεται.
    #
    # MAX_SATURATION_FOR_FIT:
    #   Παίκτης με πολύ υψηλό mean S φοράει φανέλα που ΔΕΝ ανήκει σε
    #   καμία από τις δύο κύριες ομάδες (διαιτητής neon yellow, GK bright).
    #   Εξαιρούνται από το fit για να μην τραβήξουν το K-Means μακριά
    #   από τα πραγματικά team clusters.
    #
    #   Ακαδημαϊκή σημείωση (K=2 outlier problem):
    #     Με k=2 το K-Means αναγκάζεται να βάλει ΚΑΘΕ sample σε κάποιο cluster.
    #     Αν ο διαιτητής μπει στο fit, ένα cluster κέντρο μπορεί να πηγαίνει
    #     κοντά στο χρώμα του — σπάει τη διαχωριστική ικανότητα του classifier.
    #     Στο predict (runtime) ο διαιτητής παίρνει την πιο κοντινή ομάδα
    #     σε [S, V] space — acceptable degradation για university project.
    #
    #   Γιατί 160 και όχι 140;
    #     Σκούρες navy/black φανέλες σε broadcast μπορεί να έχουν mean S
    #     μέχρι ~130-150. Threshold 140 έκοβε και legitimate dark-kit παίκτες.
    #     160 κρατάει και τα δύο kits ενώ κόβει neon yellows/reds (S>180).
    MIN_JERSEY_PIXELS      = 50   # ελάχιστα pixels μετά από grass masking
    MAX_SATURATION_FOR_FIT = 160  # διαιτητές/GKs με φωτεινές φανέλες

    def __init__(self):
        """
        Αρχικοποίηση K-Means με k=2 clusters.

        Γιατί k=2; Ένα ποδοσφαιρικό παιχνίδι έχει πάντα δύο ομάδες.
        Γιατί n_init=10; Το K-Means είναι sensitive στο αρχικό initialization.
                          10 independent trials → κρατάμε το καλύτερο (min inertia).
        Γιατί random_state=42; Reproducibility για ακαδημαϊκή αξιολόγηση.
        """
        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)

        # Flag: αποτρέπει predict_team() πριν γίνει fit
        self.is_fitted = False

    # -------------------------------------------------------------------------
    # ΒΗΜΑ 1 — FEATURE EXTRACTION
    # -------------------------------------------------------------------------

    def extract_jersey_color(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """
        Κόβει το torso του παίκτη, αφαιρεί γρασίδι με HSV mask και
        επιστρέφει 2D brightness feature vector [mean_S, mean_V].

        Γιατί [S, V] και όχι [H, S, V];
            Λευκό vs Σκούρο navy/black: το H είναι noisy για achromatic kits.
            Ο μόνος αξιόπιστος διαχωρισμός είναι η φωτεινότητα (V axis):
              Λευκό: high V (~220), low S (~15)
              Σκούρο: low V (~60),  low S (~30)

        Παράμετροι:
            frame (np.ndarray): Full BGR video frame.
            bbox  (list)      : [x1, y1, x2, y2] bounding box.

        Επιστρέφει:
            np.ndarray: [mean_S, mean_V] (float32), ή zeros αν άδειο crop.
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # --- Torso crop: 20%-50% ύψος, central 60% πλάτος ---
        #
        # Γιατί παραλείπουμε το πάνω 20%; → αποφεύγουμε κεφάλι, μαλλιά, δέρμα.
        # Γιατί σταματάμε στο 50%;          → αποφεύγουμε σορτς, κάλτσες, παπούτσια.
        # Γιατί trim 20% δεξιά/αριστερά;  → αποφεύγουμε bleeding σε γρασίδι.
        box_height = y2 - y1
        box_width  = x2 - x1

        torso_y1 = y1 + int(0.20 * box_height)
        torso_y2 = y1 + int(0.50 * box_height)
        torso_x1 = x1 + int(0.20 * box_width)
        torso_x2 = x2 - int(0.20 * box_width)

        # Guard: zero-area crop από tiny ή edge-clipped boxes
        if torso_y2 <= torso_y1 or torso_x2 <= torso_x1:
            return np.zeros(2, dtype=np.float32)

        jersey_crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]

        if jersey_crop.size == 0:
            return np.zeros(2, dtype=np.float32)

        # --- BGR → HSV (Image Basics Lab) ---
        #
        # Χρειαζόμαστε HSV τόσο για το grass mask (H space) όσο και για S, V features.
        jersey_hsv = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2HSV)

        # --- Grass mask ---
        #
        # 255 = pixel είναι γρασίδι → invert → 255 = pixel είναι φανέλα
        grass_mask  = cv2.inRange(jersey_hsv, self.GRASS_LOWER, self.GRASS_UPPER)
        jersey_mask = cv2.bitwise_not(grass_mask)

        non_grass_count = np.count_nonzero(jersey_mask)

        if non_grass_count == 0:
            return np.zeros(2, dtype=np.float32)

        # --- Mean S και V πάνω από jersey pixels μόνο ---
        #
        # cv2.mean(src, mask): μέσος κάθε channel, αγνοώντας pixels με mask=0.
        # Channel 0 = H (αγνοείται), Channel 1 = S, Channel 2 = V.
        channel_means = cv2.mean(jersey_hsv, mask=jersey_mask)
        mean_s = channel_means[1]  # Saturation
        mean_v = channel_means[2]  # Value (brightness)

        return np.array([mean_s, mean_v], dtype=np.float32)

    # -------------------------------------------------------------------------
    # ΒΗΜΑ 2 — FITTING (UNSUPERVISED LEARNING)
    # -------------------------------------------------------------------------

    def fit_teams(self, frame: np.ndarray, tracked_players: list):
        """
        Εξάγει brightness features για όλους τους παίκτες, φιλτράρει outliers
        και κάνει fit το K-Means για να ανακαλύψει τα 2 team clusters.

        Outlier filter αφαιρεί:
          - Παίκτες με λίγα jersey pixels (occluded/tiny detections).
          - Παίκτες με πολύ υψηλό mean S (διαιτητής/GK με φωτεινές φανέλες).

        Παράμετροι:
            frame           (np.ndarray): Full BGR video frame.
            tracked_players (list)      : Dicts με τουλάχιστον 'bbox' key.
        """
        clean_features  = []   # features που πέρασαν τα φίλτρα
        skipped_outlier = 0    # counter για debugging

        for player in tracked_players:
            bbox = player["bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Υπολογίζουμε ξανά το torso crop για να ελέγξουμε pixel count
            box_height = y2 - y1
            box_width  = x2 - x1
            torso_y1 = y1 + int(0.20 * box_height)
            torso_y2 = y1 + int(0.50 * box_height)
            torso_x1 = x1 + int(0.20 * box_width)
            torso_x2 = x2 - int(0.20 * box_width)

            if torso_y2 <= torso_y1 or torso_x2 <= torso_x1:
                skipped_outlier += 1
                continue

            crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
            if crop.size == 0:
                skipped_outlier += 1
                continue

            # Grass mask για να μετρήσουμε πόσα jersey pixels υπάρχουν
            crop_hsv   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            grass_mask = cv2.inRange(crop_hsv, self.GRASS_LOWER, self.GRASS_UPPER)
            jersey_mask = cv2.bitwise_not(grass_mask)
            pixel_count = np.count_nonzero(jersey_mask)

            # Φίλτρο 1: λίγα jersey pixels → noisy feature → skip
            if pixel_count < self.MIN_JERSEY_PIXELS:
                skipped_outlier += 1
                continue

            feature = self.extract_jersey_color(frame, bbox)

            # Φίλτρο 2: πολύ υψηλό S → διαιτητής/GK → skip
            mean_s = feature[0]
            if mean_s > self.MAX_SATURATION_FOR_FIT:
                skipped_outlier += 1
                continue

            clean_features.append(feature)

        print(f"[TeamClassifier] Fitting: {len(clean_features)} clean samples, "
              f"{skipped_outlier} outliers skipped.")

        if len(clean_features) < 2:
            print("[TeamClassifier] Warning: not enough clean samples to fit K-Means.")
            return

        # Stack σε (n_players, 2) matrix και fit.
        #
        # Στον 2D [S, V] χώρο:
        #   Cluster 0 → (low S, high V) = ΛΕΥΚΗ φανέλα
        #   Cluster 1 → (low S, low V)  = ΣΚΟΥΡΗ φανέλα
        # Και τα δύο kits έχουν χαμηλό S → το V είναι ο κύριος axis διαχωρισμού.
        feature_matrix = np.array(clean_features)
        self.kmeans.fit(feature_matrix)
        self.is_fitted = True

        c0 = self.kmeans.cluster_centers_[0]
        c1 = self.kmeans.cluster_centers_[1]
        print("[TeamClassifier] K-Means fitted successfully.")
        print(f"  Cluster 0 — mean S={c0[0]:.1f}, mean V={c0[1]:.1f}  "
              f"({'bright/white' if c0[1] > 130 else 'dark'})")
        print(f"  Cluster 1 — mean S={c1[0]:.1f}, mean V={c1[1]:.1f}  "
              f"({'bright/white' if c1[1] > 130 else 'dark'})")

    # -------------------------------------------------------------------------
    # ΒΗΜΑ 3 — TEAM PREDICTION
    # -------------------------------------------------------------------------

    def predict_team(self, frame: np.ndarray, bbox: list) -> int:
        """
        Προβλέπει σε ποια ομάδα (0 ή 1) ανήκει ο παίκτης.

        Κάθε άτομο στο pitch — παίκτες, διαιτητής, GKs — παίρνει label 0 ή 1.
        Δεν ξεχωρίζουμε ρητά τον διαιτητή: ο classifier θα τον βάλει στο
        πιο κοντινό cluster σε [S, V] space — acceptable για university project.

        Return values:
             0  = Team 0 (π.χ. λευκή φανέλα)
             1  = Team 1 (π.χ. σκούρη φανέλα)
            -1  = model not fitted (κάλεσε fit_teams() πρώτα)

        Παράμετροι:
            frame (np.ndarray): Full BGR video frame.
            bbox  (list)      : [x1, y1, x2, y2].

        Επιστρέφει:
            int: 0, 1, ή -1.
        """
        if not self.is_fitted:
            print("[TeamClassifier] Warning: call fit_teams() before predict_team().")
            return -1

        # Εξαγωγή [S, V] feature για αυτόν τον παίκτη
        feature = self.extract_jersey_color(frame, bbox)

        # Reshape (2,) → (1, 2): το scikit-learn θέλει 2D input array
        feature_2d = feature.reshape(1, -1)

        # Euclidean distance στον [S, V] space → index του κοντινότερου center
        team_label = int(self.kmeans.predict(feature_2d)[0])

        return team_label
