# =============================================================================
# main.py
# Κεντρικό αρχείο εκτέλεσης — επεξεργάζεται βίντεο frame-by-frame.
# Ροή: Tracker → Classifier → Possession → Visualizer → Heatmap
# (Video Basics Lab: VideoCapture loop, imshow, waitKey)
# =============================================================================

import os
import sys
import math
import cv2
import numpy as np
import ultralytics

from src.detector    import FootballDetector
from src.tracker     import FootballTracker
from src.classifier  import TeamClassifier
from src.visualizer  import FootballVisualizer
from src.ball_hough  import BallHoughDetector

# ---------------------------------------------------------------------------
# Ρυθμίσεις
# ---------------------------------------------------------------------------

VIDEO_PATH        = "data/football_match.mp4"
OUTPUT_HEATMAP_T0 = "data/heatmap_team_0.png"
OUTPUT_HEATMAP_T1 = "data/heatmap_team_1.png"

# 30 frames ≈ 1 δευτερόλεπτο — αρκεί για να δούμε και τις δύο ομάδες στο pitch
FIT_FRAMES = 30

# Κλάσεις COCO που μας ενδιαφέρουν
PLAYER_CLASS_ID = 0
BALL_CLASS_ID   = 32

# ---------------------------------------------------------------------------
# ΒΗΜΑ 0 — Άνοιγμα βίντεο (Video Basics Lab)
# ---------------------------------------------------------------------------

print("=" * 60)
print("Football Analytics — Real Video Pipeline")
print("=" * 60)

# cap.isOpened() → False αν δεν βρεθεί το αρχείο ή λείπει ο codec
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"\n[ERROR] Cannot open video file: {VIDEO_PATH}")
    print("   Make sure the file exists at the path above.")
    sys.exit(1)

# Διαβάζουμε metadata για progress tracking και αρχικοποίηση visualizer
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS)
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"\nVideo : {VIDEO_PATH}")
print(f"   Resolution : {frame_width} x {frame_height}")
print(f"   FPS        : {fps:.1f}")
print(f"   Frames     : {total_frames}")
print(f"   Duration   : {total_frames / fps:.1f} s\n")

# ---------------------------------------------------------------------------
# ΒΗΜΑ 1 — Αρχικοποίηση modules
# ---------------------------------------------------------------------------

# To μοντέλο φορτώνεται μία φορά — δεν ξαναφορτώνεται σε κάθε frame
detector   = FootballDetector()
tracker    = FootballTracker()
classifier = TeamClassifier()
visualizer = FootballVisualizer(width=frame_width, height=frame_height)

# Τρίτο επίπεδο ανίχνευσης: κλασική CV με GaussianBlur + HoughCircles
# Tier 1: ByteTrack | Tier 2: YOLO conf=0.10 | Tier 3: HoughCircles
ball_hough = BallHoughDetector()

# ---------------------------------------------------------------------------
# ΒΗΜΑ 2 — Fitting phase: μαζεύουμε χαρακτηριστικά φανέλας από τα πρώτα frames
# ---------------------------------------------------------------------------
# Χρειαζόμαστε δείγματα και από τις δύο ομάδες πριν κάνουμε fit το K-Means

print(f"Phase 1: Collecting jersey features from the first {FIT_FRAMES} frames...")

all_fit_players = []   # παίκτες που συλλέχθηκαν για fitting
fit_frame_ref   = None # θυμόμαστε το τελευταίο frame για να καλέσουμε fit_teams()

fit_frame_count = 0

while fit_frame_count < FIT_FRAMES:
    ret, frame = cap.read()

    # Τέλος βίντεο πριν συμπληρωθούν τα FIT_FRAMES
    if not ret:
        print("   Warning: video ended before fitting phase completed.")
        break

    # Παίρνουμε τα bounding boxes από τον tracker
    tracked_objects = tracker.track_frame(detector.model, frame)

    # Κρατάμε μόνο παίκτες (class 0) για το fitting
    for obj in tracked_objects:
        if obj["class_id"] == PLAYER_CLASS_ID:
            all_fit_players.append(obj)

    fit_frame_ref = frame
    fit_frame_count += 1

    if (fit_frame_count % 10) == 0 or fit_frame_count == FIT_FRAMES:
        print(f"   Fitting frame {fit_frame_count}/{FIT_FRAMES} — "
              f"players collected so far: {len(all_fit_players)}")

# Χρειαζόμαστε τουλάχιστον 2 παίκτες (έναν από κάθε ομάδα)
if len(all_fit_players) < 2:
    print("[ERROR] Not enough players detected during fitting phase. "
          "Check that the video contains visible players.")
    cap.release()
    sys.exit(1)

# Τα bboxes είναι από πολλά frames αλλά το fit_frame_ref αρκεί για εξαγωγή χρωμάτων
classifier.fit_teams(fit_frame_ref, all_fit_players)
print(f"\nK-Means fitted on {len(all_fit_players)} player samples.\n")

# ---------------------------------------------------------------------------
# ΒΗΜΑ 3 — Κύριος loop επεξεργασίας (Video Basics Lab)
# ---------------------------------------------------------------------------

print("Phase 2: Processing video — press 'q' in the display window to quit early.\n")

# Μετρητές κατοχής: πόσα frames έχει η μπάλα κοντά σε παίκτη κάθε ομάδας
frames_possession_t0 = 0
frames_possession_t1 = 0

# 80px ≈ ~0.5m στο γήπεδο — αρκετά γενναιόδωρο χωρίς να δίνουμε κατοχή από μακριά
POSSESSION_THRESHOLD_PX = 80

# Μεταβλητές μνήμης για το Φίλτρο Ταχύτητας της μπάλας
last_ball_center       = None
frames_since_last_ball = 0
MAX_BALL_SPEED_PX      = 120  # Μέγιστη λογική μετακίνηση της μπάλας ανά frame

frame_idx = fit_frame_count  # συνεχίζουμε από εκεί που σταμάτησε το fitting

while True:
    ret, frame = cap.read()

    # Τέλος βίντεο
    if not ret:
        print("\nEnd of video reached.")
        break

    frame_idx += 1

    # Ανίχνευση + Tracking μέσω ByteTrack
    tracked_objects = tracker.track_frame(detector.model, frame)

    # --- Τριεπίπεδο cascade ανίχνευσης μπάλας ---
    # Tier 1: ByteTrack (ήδη στο tracked_objects αν βρήκε μπάλα)
    # Tier 2: YOLO με conf=0.10
    # Tier 3: HoughCircles (Image Basics + Filters + Edge Detection Lab)

    ball_already_tracked = any(o["class_id"] == BALL_CLASS_ID for o in tracked_objects)

    if not ball_already_tracked:
        # Tier 2: YOLO με χαμηλό κατώφλι εμπιστοσύνης
        ball_detection = detector.detect_ball(frame, confidence_threshold=0.10)
        if ball_detection is not None:
            tracked_objects.append(ball_detection)
            ball_already_tracked = True

    if not ball_already_tracked:
        # Tier 3: HoughCircles — περνάμε το ROI mask για να αποκλείσουμε logos κλπ.
        hough_detection = ball_hough.detect(frame, roi_mask=visualizer.roi_mask)
        
        if hough_detection is not None:
            # Εξαγωγή συντεταγμένων κέντρου του υποψήφιου κύκλου Hough
            hx1, hy1, hx2, hy2 = hough_detection["bbox"]
            h_x = int((hx1 + hx2) / 2)
            h_y = int((hy1 + hy2) / 2)

            # -----------------------------------------------------------------
            # ΦΙΛΤΡΟ Α: Αποκλεισμός αν ο κύκλος είναι πάνω σε παίκτη (σορτσάκια/παπούτσια)
            # -----------------------------------------------------------------
            is_inside_player = False
            for obj in tracked_objects:
                if obj["class_id"] == PLAYER_CLASS_ID:
                    px1, py1, px2, py2 = obj["bbox"]
                    if px1 <= h_x <= px2 and py1 <= h_y <= py2:
                        is_inside_player = True
                        break

            # -----------------------------------------------------------------
            # ΦΙΛΤΡΟ Β: Έλεγχος Τηλεμεταφοράς (Velocity Distance Filter)
            # -----------------------------------------------------------------
            pass_velocity_filter = True
            if last_ball_center is not None:
                distance = math.hypot(h_x - last_ball_center[0], h_y - last_ball_center[1])
                if distance > MAX_BALL_SPEED_PX:
                    pass_velocity_filter = False

            # Η μπάλα γίνεται δεκτή στο pipeline μόνο αν περάσει και τις δύο δικλείδες
            if not is_inside_player and pass_velocity_filter:
                tracked_objects.append(hough_detection)

    # --- Κατάταξη παικτών σε ομάδες ---
    # Κάνουμε classify μία φορά εδώ και αποθηκεύουμε στο obj["team"]
    # έτσι ο visualizer απλώς διαβάζει — δεν ξαναφωνάζει τον classifier
    for obj in tracked_objects:
        if obj["class_id"] == PLAYER_CLASS_ID:
            obj["team"] = classifier.predict_team(frame, obj["bbox"])
        else:
            obj["team"] = None  # μπάλα — δεν έχει ομάδα

    # --- Υπολογισμός κατοχής μπάλας ---
    # Euclidean distance: ποιος παίκτης είναι πιο κοντά στη μπάλα;
    # Αν η απόσταση < threshold → η ομάδα του παίρνει +1 frame κατοχής

    ball_center = None
    for obj in tracked_objects:
        if obj["class_id"] == BALL_CLASS_ID:
            bx1, by1, bx2, by2 = obj["bbox"]
            ball_center = (int((bx1 + bx2) / 2), int((by2 + by1) / 2))
            break  # μία μπάλα μόνο

    if ball_center is not None:
        # Ενημέρωση μνήμης κίνησης: Βρέθηκε έγκυρη μπάλα
        last_ball_center = ball_center
        frames_since_last_ball = 0

        min_distance  = float("inf")
        closest_team  = -1

        for obj in tracked_objects:
            if obj["class_id"] != PLAYER_CLASS_ID:
                continue
            team = obj.get("team", -1)
            if team not in (0, 1):
                continue  # αταξινόμητος ή μπάλα

            # Χρησιμοποιούμε το πόδι (κάτω-κεντρικό bbox) — εκεί αγγίζει το έδαφος
            bx1, by1, bx2, by2 = obj["bbox"]
            foot_x = int((bx1 + bx2) / 2)
            foot_y = int(by2)

            # math.hypot: sqrt(dx²+dy²) χωρίς numpy εξάρτηση
            dx = ball_center[0] - foot_x
            dy = ball_center[1] - foot_y
            distance = math.hypot(dx, dy)

            if distance < min_distance:
                min_distance = distance
                closest_team = team

        # Αποδίδουμε κατοχή μόνο αν ο παίκτης είναι αρκετά κοντά
        if min_distance < POSSESSION_THRESHOLD_PX:
            if closest_team == 0:
                frames_possession_t0 += 1
            elif closest_team == 1:
                frames_possession_t1 += 1
    else:
        # Μηχανισμός Γέφυρας: Αν χαθεί η μπάλα, κράτα τη μνήμη της για 3 frames πριν μηδενίσεις
        frames_since_last_ball += 1
        if frames_since_last_ball > 3:
            last_ball_center = None

    # --- Visualization ---
    annotated_frame = visualizer.draw_predictions(frame, tracked_objects)

    # Possession bar με cv2.addWeighted (Filters Lab)
    annotated_frame = visualizer.draw_possession_overlay(
        annotated_frame, frames_possession_t0, frames_possession_t1
    )

    # Ενημέρωση heatmap accumulators
    visualizer.update_heatmap(tracked_objects)

    # Εμφάνιση frame (Video Basics Lab)
    cv2.imshow("Football Analytics Demo", annotated_frame)

    # waitKey(1): 1ms αναμονή — απαραίτητο για να "ζωντανεύει" το παράθυρο
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n'q' pressed — stopping early.")
        break

    if frame_idx % 100 == 0:
        progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
        print(f"   Frame {frame_idx}/{total_frames}  ({progress:.1f}%)")

# ---------------------------------------------------------------------------
# ΒΗΜΑ 4 — Cleanup και αποθήκευση αποτελεσμάτων
# ---------------------------------------------------------------------------

# Πάντα release — αν δεν το κάνουμε μπορεί να χαλάσει το αρχείο
cap.release()
cv2.destroyAllWindows()

# Στατιστικά κατοχής
total_possession = frames_possession_t0 + frames_possession_t1
if total_possession > 0:
    final_pct_t0 = (frames_possession_t0 / total_possession) * 100
    final_pct_t1 = (frames_possession_t1 / total_possession) * 100
else:
    final_pct_t0 = final_pct_t1 = 0.0

print("\n--- Final Possession Statistics ---")
print(f"   Team 0 (White / Paris FC) : {frames_possession_t0:>5} frames  ({final_pct_t0:.1f}%)")
print(f"   Team 1 (Dark  / PSG)      : {frames_possession_t1:>5} frames  ({final_pct_t1:.1f}%)")
print(f"   Total attributed frames   : {total_possession}")
print(f"   Total video frames        : {frame_idx}")
print(f"   Ball detection rate       : {(total_possession / max(frame_idx, 1)) * 100:.1f}%")
print()
print("   Ball Detection Chain (three-tier fallback system with noise suppression):")
print("      Tier 1 — ByteTrack       : persistent track_id, highest reliability")
print("      Tier 2 — YOLO detect_ball: deep learning, conf >= 0.10")
print("      Tier 3 — HoughCircles    : classical CV + BBox Exclusion Filter & Velocity Verification")
print("      Each tier activates only if the previous one found nothing in that frame.")

print("\nGenerating team-specific tactical heatmaps...")

# Δύο ξεχωριστά heatmaps — ένα ανά ομάδα για τακτική ανάλυση
os.makedirs("data", exist_ok=True)

for team_id, output_path in [(0, OUTPUT_HEATMAP_T0), (1, OUTPUT_HEATMAP_T1)]:
    heatmap_image = visualizer.generate_team_heatmap(team_id)
    success = cv2.imwrite(output_path, heatmap_image)
    if success:
        print(f"   Team {team_id} heatmap saved -> {output_path}  "
              f"({heatmap_image.shape[1]}x{heatmap_image.shape[0]} px)")
    else:
        print(f"   [ERROR] Could not write Team {team_id} heatmap to {output_path}")

print("\n" + "=" * 60)
print(f"[OK] Pipeline complete. Processed {frame_idx} frames total.")
print(f"     Heatmaps: {OUTPUT_HEATMAP_T0}")
print(f"               {OUTPUT_HEATMAP_T1}")
print("=" * 60)
