import pygame
import sys
import cv2
import math
from ultralytics import YOLO
import numpy as np
import time
from session import save_session, get_total_pushups, get_total_squats

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
    return pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), 'RGB')

# ---------- Trainer Logic ----------
def run_trainer(screen, clock, exercise, target_reps, window_size=(900,600)):
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
    card_x, card_y = (W-card_w)//2, H//2-card_h//2

    font_big = pygame.font.SysFont("Arial", 36, bold=True)
    font_medium = pygame.font.SysFont("Arial", 24)
    font_small = pygame.font.SysFont("Arial", 20)

    # Calories mapping
    calories_map = {5: 5, 10: 10, 15: 15}
    calories_burned = calories_map.get(target_reps, 0)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    cap.release()
                    return
                if level_complete and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    save_session(counter, exercise=exercise)
                    cap.release()
                    return

        ret, frame = cap.read()
        if not ret:
            continue

        # Resize frame and initialize annotated
        frame_proc = cv2.resize(frame, (640, int(frame.shape[0]*640/frame.shape[1])))
        annotated = frame_proc.copy()
        person_found = False

        # YOLO pose detection
        results = model(frame_proc, verbose=False)
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

            # Push-up detection
            if exercise == "pushup" and kps.shape[0] >= 10:
                shoulder, elbow, wrist = tuple(kps[5]), tuple(kps[7]), tuple(kps[9])
                angle_val = calculate_angle(shoulder, elbow, wrist)
                angle_history.append(angle_val)
                if len(angle_history) > 8: angle_history.pop(0)
                angle_val = sum(angle_history)/len(angle_history)

                if angle_val > 135:
                    stage = "up"
                if angle_val < 120 and stage == "up":
                    stage = "down"
                    counter += 1

            # Squat detection
            elif exercise == "squat" and kps.shape[0] >= 16:
                hip, knee, ankle = tuple(kps[11]), tuple(kps[13]), tuple(kps[15])
                angle_val = calculate_angle(hip, knee, ankle)
                angle_history.append(angle_val)
                if len(angle_history) > 8: angle_history.pop(0)
                angle_val = sum(angle_history)/len(angle_history)

                if angle_val > 150:
                    stage = "up"
                if angle_val < 120 and stage == "up":
                    stage = "down"
                    counter += 1

        # Draw camera
        surf = cv2frame_to_pygame_surface(annotated)
        cam_w, cam_h = surf.get_width(), surf.get_height()
        scale = W / cam_w
        cam_surf = pygame.transform.smoothscale(surf, (int(cam_w*scale), int(cam_h*scale)))
        screen.fill((10,10,10))
        screen.blit(cam_surf, (0,0))

        # Info card
        card_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
        pygame.draw.rect(card_surf, (30,30,30,220), (0,0,card_w,card_h), border_radius=16)
        card_surf.blit(font_big.render(f"Level: {target_reps} reps", True, (245,245,245)), (16,10))
        card_surf.blit(font_medium.render(f"{counter}/{target_reps}", True, (200,255,200)), (16,60))
        card_surf.blit(font_small.render(f"Angle: {int(angle_val)}°", True, (200,200,255)), (16,96))
        screen.blit(card_surf, (card_x, card_y))

        # Level complete overlay
        if counter >= target_reps:
            level_complete = True
            complete_time = complete_time or time.time()
            overlay = pygame.Surface((W,H), pygame.SRCALPHA)
            overlay.fill((0,0,0,140))
            screen.blit(overlay, (0,0))

            big = pygame.font.SysFont("Arial", 56, bold=True)
            screen.blit(big.render("LEVEL COMPLETE!", True, (180,255,180)),
                        big.render("LEVEL COMPLETE!", True, (180,255,180)).get_rect(center=(W//2,H//2-60)))

            # Calories burned
            screen.blit(font_big.render(f"Calories Burned: {calories_burned}", True, (255,200,100)),
                        font_big.render(f"Calories Burned: {calories_burned}", True, (255,200,100)).get_rect(center=(W//2,H//2)))

            screen.blit(font_small.render("Press Enter to return to menu", True, (220,220,220)),
                        font_small.render("Press Enter to return to menu", True, (220,220,220)).get_rect(center=(W//2,H//2+60)))

        if not person_found and not level_complete:
            hint = font_small.render("No person detected — move into the frame", True, (220,180,180))
            screen.blit(hint, (20,H-40))

        pygame.display.flip()
        clock.tick(30)

        # Auto-save after 10 seconds
        if level_complete and complete_time and time.time() - complete_time > 10:
            save_session(counter, exercise=exercise)
            cap.release()
            return

# ---------- Level Selection ----------
def choose_level(screen, clock, exercise):
    W, H = 900, 600
    font_btn = pygame.font.SysFont("Arial", 26)
    font_title = pygame.font.SysFont("Arial", 36, bold=True)
    font_small = pygame.font.SysFont("Arial", 20)
    BLUE, WHITE, BG = (60,140,255), (245,245,245), (18,18,18)

    btn_w, btn_h, gap = 200, 60, 40
    levels = [("Level 1",5),("Level 2",10),("Level 3",15)]
    buttons = [ (pygame.Rect(W//2-btn_w//2,200+i*(btn_h+gap),btn_w,btn_h),text,reps)
                for i,(text,reps) in enumerate(levels) ]

    while True:
        screen.fill(BG)
        screen.blit(font_title.render(f"Select Level for {exercise.title()}", True, WHITE),
                    font_title.render(f"Select Level for {exercise.title()}", True, WHITE).get_rect(center=(W//2,100)))

        for rect,text,_ in buttons:
            pygame.draw.rect(screen, BLUE, rect, border_radius=16)
            screen.blit(font_btn.render(text, True, WHITE),
                        font_btn.render(text, True, WHITE).get_rect(center=rect.center))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                for rect,_,reps in buttons:
                    if rect.collidepoint(pos):
                        run_trainer(screen, clock, exercise, reps, window_size=(W,H))
                        return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return

        pygame.display.flip()
        clock.tick(30)

# ---------- Main Menu ----------
def main():
    pygame.init()
    W, H = 900, 600
    screen = pygame.display.set_mode((W,H))
    pygame.display.set_caption("Body Trainer")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("Arial",48,bold=True)
    font_btn = pygame.font.SysFont("Arial",26)
    font_stats = pygame.font.SysFont("Arial",22)
    WHITE,BG,BLUE = (245,245,245),(18,18,18),(60,140,255)

    btn_w, btn_h = 200,70
    btn_pushups = pygame.Rect(W//2-btn_w-40,200,btn_w,btn_h)
    btn_squats  = pygame.Rect(W//2+40,200,btn_w,btn_h)

    while True:
        screen.fill(BG)
        screen.blit(font_title.render("Body Trainer", True, WHITE),
                    font_title.render("Body Trainer", True, WHITE).get_rect(center=(W//2,70)))

        # Stats
        screen.blit(font_stats.render(f"Total Push-ups: {get_total_pushups()}", True, (200,255,200)),
                    (btn_pushups.x+btn_w//2-90,btn_pushups.y-30))
        screen.blit(font_stats.render(f"Total Squats: {get_total_squats()}", True, (200,255,200)),
                    (btn_squats.x+btn_w//2-70,btn_squats.y-30))

        # Buttons
        pygame.draw.rect(screen, BLUE, btn_pushups, border_radius=16)
        pygame.draw.rect(screen, BLUE, btn_squats, border_radius=16)
        screen.blit(font_btn.render("Push Ups", True, WHITE),
                    font_btn.render("Push Ups", True, WHITE).get_rect(center=btn_pushups.center))
        screen.blit(font_btn.render("Squat", True, WHITE),
                    font_btn.render("Squat", True, WHITE).get_rect(center=btn_squats.center))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                if btn_pushups.collidepoint(pos): choose_level(screen, clock, "pushup")
                if btn_squats.collidepoint(pos):  choose_level(screen, clock, "squat")
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()
