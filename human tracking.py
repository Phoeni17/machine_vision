import pygame
import sys
import cv2
import math
from ultralytics import YOLO
import numpy as np
import time
from session import save_session, get_total_pushups

# ---------- Utilities ----------
def calculate_angle(a, b, c):
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def cv2frame_to_pygame_surface(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    surface = pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), 'RGB')
    return surface

# ---------- YOLO push-up counter ----------
def run_pushup_counter_in_pygame(screen, clock, target_reps, window_size=(900, 600)):
    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open camera")
        return

    counter = 0
    stage = None
    angle_val = 0
    level_complete = False
    complete_time = None
    angle_history = []

    W, H = window_size

    card_w, card_h = 320, 160
    card_x = (W - card_w) // 2
    card_y = H // 2 - card_h // 2

    font_big = pygame.font.SysFont("Arial", 36, bold=True)
    font_small = pygame.font.SysFont("Arial", 20)
    font_medium = pygame.font.SysFont("Arial", 24)

    # -------------------
    # LEVEL LOOP
    # -------------------
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                cv2.destroyAllWindows()
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                if level_complete and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    # ⭐ SAVE SESSION HERE ⭐
                    save_session(counter)

                    cap.release()
                    cv2.destroyAllWindows()
                    return

        ret, frame = cap.read()
        if not ret:
            print("Camera frame missing.")
            cap.release()
            return

        scale_width = 640
        h, w = frame.shape[:2]
        if w != scale_width:
            ratio = scale_width / float(w)
            new_h = int(h * ratio)
            frame_proc = cv2.resize(frame, (scale_width, new_h))
        else:
            frame_proc = frame

        # YOLO inference
        results = model(frame_proc, verbose=False)
        annotated = frame_proc.copy()

        person_found = False
        for r in results:
            keypoints = getattr(r, 'keypoints', None)
            if keypoints is None or getattr(keypoints, 'xy', None) is None or keypoints.xy.shape[0] == 0:
                continue

            person_found = True
            try:
                annotated = r.plot()
            except:
                annotated = frame_proc.copy()

            try:
                kps = keypoints.xy[0].cpu().numpy()
            except:
                continue

            if kps.shape[0] >= 10:
                shoulder = tuple(kps[5])
                elbow = tuple(kps[7])
                wrist = tuple(kps[9])

                angle_val = calculate_angle(shoulder, elbow, wrist)
                angle_history.append(angle_val)
                if len(angle_history) > 8:
                    angle_history.pop(0)
                angle_val = sum(angle_history) / len(angle_history)

                if angle_val > 160:
                    stage = "up"
                if angle_val < 90 and stage == "up":
                    stage = "down"
                    counter += 1
                    print("Count:", counter)

        # Draw camera
        surf = cv2frame_to_pygame_surface(annotated)
        cam_w, cam_h = surf.get_width(), surf.get_height()
        scale = W / cam_w
        new_cam_w = int(cam_w * scale)
        new_cam_h = int(cam_h * scale)
        cam_surf = pygame.transform.smoothscale(surf, (new_cam_w, new_cam_h))

        screen.fill((10, 10, 10))
        cam_x = (W - new_cam_w) // 2
        cam_y = 0
        screen.blit(cam_surf, (cam_x, cam_y))

        # Card
        card_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
        card_surf.fill((20, 20, 20, 200))
        pygame.draw.rect(card_surf, (30, 30, 30, 220), (0, 0, card_w, card_h), border_radius=16)

        card_surf.blit(font_big.render(f"Level: {target_reps} reps", True, (245, 245, 245)), (16, 10))
        card_surf.blit(font_medium.render(f"{int(counter)}/{target_reps}", True, (200, 255, 200)), (16, 60))
        card_surf.blit(font_small.render(f"Elbow: {int(angle_val)}°", True, (200, 200, 255)), (16, 96))

        pb_x, pb_y, pb_w, pb_h = 16, 120, card_w - 32, 22
        pygame.draw.rect(card_surf, (60, 60, 60, 180), (pb_x, pb_y, pb_w, pb_h), border_radius=12)

        progress = min(1.0, counter / max(1, target_reps))
        fill_w = int(pb_w * progress)
        if fill_w > 0:
            pygame.draw.rect(card_surf, (80, 220, 100, 220), (pb_x, pb_y, fill_w, pb_h), border_radius=12)

        screen.blit(card_surf, (card_x, card_y))

        # Level complete overlay
        if counter >= target_reps:
            level_complete = True
            complete_time = complete_time or time.time()

            overlay = pygame.Surface((W, H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            screen.blit(overlay, (0, 0))

            big = pygame.font.SysFont("Arial", 56, bold=True)
            msg = big.render("LEVEL COMPLETE!", True, (180, 255, 180))
            screen.blit(msg, msg.get_rect(center=(W//2, H//2 - 30)))

            sub = font_small.render("Press Enter to return to menu", True, (220, 220, 220))
            screen.blit(sub, sub.get_rect(center=(W//2, H//2 + 30)))

        if not person_found and not level_complete:
            hint = font_small.render("No person detected — move into the frame", True, (220, 180, 180))
            screen.blit(hint, (20, H - 40))

        pygame.display.flip()
        clock.tick(30)

        # Auto exit after 10s
        if level_complete and complete_time and time.time() - complete_time > 10:
            save_session(counter)
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()


# ---------- Main Menu ----------
def main():
    pygame.init()
    W, H = 900, 600
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Push-up Trainer (Floating Card UI)")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("Arial", 48, bold=True)
    font_btn = pygame.font.SysFont("Arial", 26)
    font_stats = pygame.font.SysFont("Arial", 22)
    WHITE = (245, 245, 245)
    BG = (18, 18, 18)
    BLUE = (60, 140, 255)

    btn_w, btn_h = 360, 70
    btn1 = pygame.Rect((W - btn_w)//2, 170, btn_w, btn_h)
    btn2 = pygame.Rect((W - btn_w)//2, 260, btn_w, btn_h)
    btn3 = pygame.Rect((W - btn_w)//2, 350, btn_w, btn_h)

    while True:
        screen.fill(BG)
        title = font_title.render("Push-up Counter", True, WHITE)
        screen.blit(title, title.get_rect(center=(W//2, 70)))

        # ⭐ SHOW TOTAL PUSHUPS ⭐
        total_pushups = get_total_pushups()
        stats_text = font_stats.render(f"Total pushups ever: {total_pushups}", True, (200, 255, 200))
        screen.blit(stats_text, (W//2 - 170, 120))

        pygame.draw.rect(screen, BLUE, btn1, border_radius=16)
        pygame.draw.rect(screen, BLUE, btn2, border_radius=16)
        pygame.draw.rect(screen, BLUE, btn3, border_radius=16)

        screen.blit(font_btn.render("Level 1 — 10 reps", True, WHITE),
                    font_btn.render("Level 1 — 10 reps", True, WHITE).get_rect(center=btn1.center))

        screen.blit(font_btn.render("Level 2 — 20 reps", True, WHITE),
                    font_btn.render("Level 2 — 20 reps", True, WHITE).get_rect(center=btn2.center))

        screen.blit(font_btn.render("Level 3 — 30 reps", True, WHITE),
                    font_btn.render("Level 3 — 30 reps", True, WHITE).get_rect(center=btn3.center))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                if btn1.collidepoint(pos):
                    run_pushup_counter_in_pygame(screen, clock, target_reps=10, window_size=(W, H))
                if btn2.collidepoint(pos):
                    run_pushup_counter_in_pygame(screen, clock, target_reps=20, window_size=(W, H))
                if btn3.collidepoint(pos):
                    run_pushup_counter_in_pygame(screen, clock, target_reps=30, window_size=(W, H))
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()
