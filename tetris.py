import pygame
import random
import math, time, threading, queue, argparse, ctypes
import numpy as np
from datetime import datetime
import cv2
from picamera2 import Picamera2
import mediapipe as mp
import speech_recognition as sr
from pose_detect import classifyPose
import ctypes
from tetris_net import create_link
from attack import grid_indices_from_landmarks

ctypes.cdll.LoadLibrary("libasound.so").snd_lib_error_set_handler(
ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)(lambda *_: None))

a = argparse.ArgumentParser()
a.add_argument("--mic", type=int, default=None, help="device index (list & auto if omitted)")
a.add_argument("--dual", nargs="+", metavar=("mode", "ip"),
               help="host ︱ peer <host_ip> ︱ 不給 = 單人")
a.add_argument("--no-cam", action="store_true",
               help="skip camera init (use black frame)")
args = a.parse_args()



# print("[INFO] 可用錄音裝置：")
for i, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"  {i} : {name}")

if args.mic is None:
    for i, name in enumerate(sr.Microphone.list_microphone_names()):
        try:
            with sr.Microphone(device_index=i) as _:
                args.mic = i; break
        except Exception:
            continue
    # print(f"[INFO] 自動選擇 device_index = {args.mic}")


colors = [
    (0, 0, 0),
    (120, 37, 179),
    (100, 179, 179),
    (80, 34, 22),
    (80, 134, 22),
    (180, 34, 22),
    (180, 34, 122),
    (180, 134, 122),
    (0, 0, 0) 
]


    # block shapes
    # 0  1  2  3  
    # 4  5  6  7  
    # 8  9 10 11  
    # 12 13 14 15
figures_label = ["0", "I", "right S", "left S", "right L", "left L", "T", "O"]
figures = [
    [[]],
    [[1, 5, 9, 13], [4, 5, 6, 7]], # 長方形
    [[4, 5, 9, 10], [2, 6, 5, 9]], # 右上左下Ｓ形
    [[5, 6, 8, 9], [1, 5, 6, 10]], # 左上右下S形
    [[1, 2, 5, 9], [4, 5, 6, 10], [1, 5, 9, 8], [0, 4, 5, 6]], # 倒7形
    [[1, 2, 6, 10], [3, 5, 6, 7], [2, 6, 10, 11], [5, 6, 7, 9]], # 7形
    [[1, 4, 5, 6], [1, 5, 6, 9], [4, 5, 6, 9], [1, 4, 5, 9]], # T形
    [[5, 6, 9, 10]], # 正方形
]

L_TRIG = {"left", "左"}
R_TRIG = {"right", "右"}
S_TRIG = {"spin", "轉"}
D_TRIG = {"down", "下"}

def clear_queue(q: queue.Queue):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break

def voice_thread(q: queue.Queue, txt_q: queue.Queue, dev_idx):
    rec = sr.Recognizer()
    with sr.Microphone(device_index=dev_idx) as src:
        rec.adjust_for_ambient_noise(src, duration=1.5)
        print("[INFO] Mic calibrated, start listening…")
        while True:
            audio = rec.listen(src, phrase_time_limit=1.5)
            try:
                txt = rec.recognize_google(audio, language="zh-TW").lower()
                txt_q.put(txt)
                print("[HEARD]", txt)
                if any(w in txt for w in L_TRIG):
                    q.put("left")
                elif any(w in txt for w in R_TRIG):
                    q.put("right")
                elif any(w in txt for w in S_TRIG):
                    q.put("rotate")
                elif any(w in txt for w in D_TRIG):
                    q.put("down")
                    
            except sr.UnknownValueError:
                pass
            #     print("[HEARD] <unrecognized>")
            except sr.RequestError as e:
                print("[AUDIO]", e)

# Block class
class Figure:
    x = 0
    y = 0


    def __init__(self, x, y, select_type):
        self.x = x
        self.y = y
        self.type = select_type
        self.color = self.type
        self.rotation = 0

    def image(self):
        return figures[self.type][self.rotation]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(figures[self.type])


class Tetris:
    def __init__(self, height, width):
        self.level = 0.7
        self.score = 0
        self.state = "start"
        self.field = []
        self.height = 0
        self.width = 0
        self.x = 100
        self.y = 100
        self.zoom = 25 # rect size
        self.figure = None
        self.combo3 = 0
        self.attack_combo = 2
        self.attacking = False

        self.select_type = None
        self.height = height
        self.width = width
        self.field = []
        self.score = 0
        self.state = "start"
        for _ in range(height):
            new_line = []
            for _ in range(width):
                new_line.append(0)
            self.field.append(new_line)

    def new_figure(self, select_type):
        self.figure = Figure(3, 0, select_type)

    def intersects(self):
        intersection = False
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    if i + self.figure.y > self.height - 1 or \
                            j + self.figure.x > self.width - 1 or \
                            j + self.figure.x < 0 or \
                            self.field[i + self.figure.y][j + self.figure.x] > 0:
                        intersection = True
        return intersection

    def break_lines(self):
        lines = 0
        for i in range(1, self.height):
            zeros = 0
            for j in range(self.width):
                if self.field[i][j] == 0:
                    zeros += 1
            if zeros == 0:
                lines += 1
                for i1 in range(i, 1, -1):
                    for j in range(self.width):
                        self.field[i1][j] = self.field[i1 - 1][j]
        self.score += lines ** 2
        return lines

    def go_space(self):
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        self.freeze()

    def go_down(self):
        self.figure.y += 1
        if self.intersects():
            self.figure.y -= 1
            self.freeze()

    def freeze(self):
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    self.field[i + self.figure.y][j + self.figure.x] = self.figure.color
        lines = self.break_lines()
        self.combo3 += lines
        if lines > 0:
            print(f"[COMBO] 連擊 {self.combo3} 次，再加 {self.attack_combo - self.combo3} 行即可攻擊！")

        # send attack
        if self.combo3 >= self.attack_combo:
            self.attacking = True
            print(f"[COMBO] 連擊 {self.combo3} 次")
            occ = None
            if cam is not None:  
                i = 0                       # 只有有相機才抓
                while(True):   
                    i+=1                  # 最多嘗試 30 幀 (~1 秒)
                    print(f"[ATTACK] 嘗試抓取九宮格…{i}")
                    frame = cam.capture_array() if isinstance(cam, Picamera2) else cam.read()[1]
                    rgb   = frame if frame.shape[2]==3 else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res   = pose.process(rgb)
                    if res.pose_landmarks:              # 偵測到骨架
                        h,w = rgb.shape[:2]
                        occ = grid_indices_from_landmarks(h, w, res.pose_landmarks.landmark)
                        if len(occ) > 0:              # 有抓到九宮格
                            occ = sorted(occ)
                            # print(f"[ATTACK] 抓到九宮格：{occ}")
                            if recv_q:                  # 有開 --dual 時才送
                                net_send({"type": "attack", "occ": occ})
                            else:                       # 單人模式自玩
                                self.get_attack(occ)
                            break
                    time.sleep(0.03)
            
            else:
                print("[ATTACK] 單人模式無相機，跳過攻擊") 
            self.combo3 = 0
            self.attacking = False
        self.new_figure(1)
        self.select_type = None

        if self.intersects():
            self.state = "gameover"
        self.figure = None

    def go_side(self, dx):
        old_x = self.figure.x
        self.figure.x += dx
        if self.intersects():
            self.figure.x = old_x

    def rotate(self):
        old_rotation = self.figure.rotation
        self.figure.rotate()
        if self.intersects():
            self.figure.rotation = old_rotation

    def get_attack(self, occ_indices: list[int]):
        """隨機水平 & 下落檢查的灰色攻擊塊 (九宮格 3×3)"""
        max_try = 10
        shape_w = 3          # 九宮格寬度
        shape_h = 3
        base_rows = [0,1,2]  # row 0..2 映射到 occ row 0..2

        for _ in range(max_try):
            ox = random.randint(0, self.width - shape_w)  # 隨機水平起點

            # ---------- 找可落到底的 y ----------
            y = 0
            while True:
                collide = False
                for idx in occ_indices:
                    r, c = divmod(idx, 3)
                    gx = ox + c
                    gy = y + r
                    # 超下邊或撞現有方塊就算碰撞
                    if gy >= self.height or self.field[gy][gx] > 0:
                        collide = True
                        break
                if collide:
                    y -= 1      # 上一步才是合法
                    break
                y += 1
            # y 可能變成 -1 -> 放不下
            if y < 0:
                continue        # 換下一個 ox 再試
            # ---------- 寫入棋盤 ----------
            for idx in occ_indices:
                r, c = divmod(idx, 3)
                gx = ox + c
                gy = y + r
                self.field[gy][gx] = 8 
            # print(f'[ATTACK] 灰色攻擊塊落在 x={ox}, y={y}')
            return

        # print('[ATTACK] 場上擁擠，直接 Game Over')
        self.state = "gameover"


def draw_grid(pygame, screen, game):
    shift = 20
    pygame.draw.rect(screen, WHITE, [game.x-shift, game.y-shift, game.zoom*game.width+shift*2, game.zoom*game.height+shift*2], border_radius=25)
    pygame.draw.rect(screen, (50, 50, 50), [game.x-shift, game.y-shift, game.zoom*game.width+shift*2, game.zoom*game.height+shift*2], border_radius=25, width=10)

    for i in range(game.height):
        for j in range(game.width):
            pygame.draw.rect(screen, GRAY, [game.x + game.zoom * j, game.y + game.zoom * i, game.zoom, game.zoom], 1)
            if game.field[i][j] > 0:
                pygame.draw.rect(screen, colors[game.field[i][j]],
                                [game.x + game.zoom * j + 1, game.y + game.zoom * i + 1, game.zoom - 2, game.zoom - 1])
                
def draw_selection(pygame, screen, types):

    pygame.draw.rect(screen, WHITE, [720, 30, 350, 190], border_radius=25)
    pygame.draw.rect(screen, (50, 50, 50), [720, 30, 350, 190], border_radius=25, width=1)
    pygame.draw.line(screen, (50, 50, 50), [720, 80], [1070, 80], width=1)
    font = pygame.font.SysFont('Calibri', 40, True, False)
    text = font.render("Pose It !", True, (50, 50, 50))
    screen.blit(text, [830, 40])

    for t in range(len(types)):
        type = types[t]
        for i in range(4):
            for j in range(4):
                p = i * 4 + j
                if p in figures[type][0]:
                    pygame.draw.rect(screen, colors[type],
                                    [game.x + game.zoom * (j + game.width + 4*t) + 400,
                                    game.y + game.zoom * (i) + 1,
                                    game.zoom - 2, game.zoom - 2])
                    
# select_type = None
# chosen = False

if __name__ == "__main__":

    if args.dual:
        mode = args.dual[0]
        peer_ip = None if mode == "host" else args.dual[1]
        print(f"[INFO] {mode}模式，對手 IP：{peer_ip if peer_ip else '無'}")
        recv_q, net_send = create_link(mode, peer_ip)
    else:
        print("[INFO] 單人模式，無網路連線")
        recv_q, net_send = None, lambda *_: None 
    # ──────────── Mediapipe Pose ────────────
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=0,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # ──────────── Camera ────────────
    W,H = 640,480
    if args.no_cam:
        cam = None
        cmd_q = queue.Queue()    # 仍給空 Queue，程式不會阻塞
        txt_q = queue.Queue()    # 語音指令 Queue
    else:
        cam = Picamera2(); cam.configure(cam.create_video_configuration(main={"size":(W,H),"format":"BGR888"})); cam.start()
        cmd_q = queue.Queue()
        txt_q = queue.Queue()
        threading.Thread(target=voice_thread, args=(cmd_q,txt_q,args.mic), daemon=True).start()


    label = 0; t0=0; prev_label = 0; candidate = 0
    # Initialize the game engine
    pygame.init()
    pygame.key.set_repeat(250, 30)

    # Define some colorsx
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128)

    size = (1500, 900)
    screen = pygame.display.set_mode(size)
    bg_image = pygame.image.load("src/background.png").convert_alpha()
    bg_image = pygame.transform.scale(bg_image, (1500, 900))
    bright_layer = pygame.Surface(bg_image.get_size()).convert_alpha()
    bright_layer.fill((255, 255, 255, 60))
    bg_image.set_alpha(80)

    pygame.display.set_caption("Tetris")

    # Loop until the user clicks the close button.
    done = False
    clock = pygame.time.Clock()
    fps = 25
    screen_width = 10
    screen_height = 30
    game = Tetris(screen_height, screen_width)
    counter = 0

    pressing_down = False

    types = random.sample(range(1, len(figures)), 3)
    txt = ""
    while not done:
        if recv_q:
            while not recv_q.empty():
                m = recv_q.get()
                if m["type"] == "attack":
                    game.get_attack(m["occ"])
                elif m["type"] == "gameover":
                    game.state = "gameover"
        screen.fill(WHITE)

        screen.blit(bg_image, [0,0])
        screen.blit(bright_layer, [0,0])

        if cam is None:
            frame = np.zeros((H, W, 3), dtype=np.uint8)
            rgb = frame
        else:
            rgb = cam.capture_array()
        results = pose.process(rgb)
        label = 0
        text_label = "Unknown"
        if results.pose_landmarks:
            # draw skeleton
            mp_drawing.draw_landmarks(
                rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            # collect landmarks in pixel coords
            h, w = rgb.shape[:2]
            lm = [
                (int(l.x * w), int(l.y * h), l.visibility)
                for l in results.pose_landmarks.landmark
            ]
            label = classifyPose(lm)
            text_label = figures_label[label]
        if label is not 0 and  game.select_type is None:
            if label != candidate:
                candidate = label
                start_time = time.time()
            elif time.time() - start_time >= 1.0 and candidate != prev_label:
                prev_label = candidate
                # print(f"[LABEL] {datetime.now().strftime('%H:%M:%S')} → {candidate}:{figures_label[candidate]}")
                if candidate != "0" and candidate in types:
                   game.select_type = candidate


        # overlay label
        cv2.putText(rgb, figures_label[label], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        surface = pygame.surfarray.make_surface(np.rot90(rgb))
        # surface = pygame.surfarray.make_surface(rgb)
        screen.blit(surface, (570, 250))
        # cv2.imshow("Live", rgb)

        if cv2.waitKey(1)&0xFF==ord('q'):
            break
        if game.select_type is None:
            draw_grid(pygame, screen, game)
            draw_selection(pygame, screen, types)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        game.select_type = 0
                    if event.key == pygame.K_2:
                        game.select_type = 1
                    if event.key == pygame.K_3:
                        game.select_type = 2
            if game.select_type is not None:
                game.select_type = types[game.select_type]
        else:
            if game.figure is None:
                game.new_figure(game.select_type)
                types = random.sample(range(1, len(figures)), 3)
            counter += 1
            if counter > 100000:
                counter = 0

            if counter % (fps // game.level // 2) == 0 or pressing_down:
                if game.state == "start":
                    # pass
                    game.go_down()

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        game.rotate()
                    if event.key == pygame.K_DOWN:
                        pressing_down = True
                    if event.key == pygame.K_LEFT:
                        game.go_side(-1)
                    if event.key == pygame.K_RIGHT:
                        game.go_side(1)
                    if event.key == pygame.K_SPACE:
                        game.go_space()
                    if event.key == pygame.K_ESCAPE:
                        game.__init__(screen_height, screen_width)

            if event.type == pygame.KEYUP:
                    if event.key == pygame.K_DOWN:
                        pressing_down = False

            if game.select_type is not None:
                txt = txt_q.get() if not txt_q.empty() else txt
                clear_queue(txt_q)
                
            if game.select_type is not None and not cmd_q.empty():
                cmd = cmd_q.get()
                clear_queue(cmd_q)
                if cmd=="left":
                    game.go_side(-1)
                elif cmd=="right":
                    game.go_side(1)
                elif cmd=="down":
                    game.go_space()
                elif cmd=="rotate":
                    game.rotate()

            # Draw grid field
            draw_grid(pygame, screen, game)

            if game.figure is not None:
                shadow_y = game.figure.y
                bottom = screen_height - (max(game.figure.image()) // 4) - 1
                collapse = False
                while shadow_y < bottom and (not collapse):
                    shadow_y += 1
                    for p in game.figure.image():
                        i = p // 4
                        j = p % 4
                        x = j + game.figure.x
                        y = i + shadow_y
                        if game.field[y][x] > 0:
                            collapse = True
                            shadow_y -= 1
                            break

                for i in range(4):
                    for j in range(4):
                        p = i * 4 + j
                        if p in game.figure.image():
                            # Draw current block shadow
                            pygame.draw.rect(screen, (150, 150, 150),
                                            [game.x + game.zoom * (j + game.figure.x) + 1,
                                            # game.y + game.zoom * (i + game.figure.y + 10) + 1,
                                            game.y + game.zoom * (i + shadow_y) + 1,
                                            game.zoom - 2, game.zoom - 2])
                            
                            # Draw current block
                            pygame.draw.rect(screen, colors[game.figure.color],
                                            [game.x + game.zoom * (j + game.figure.x) + 1,
                                            game.y + game.zoom * (i + game.figure.y) + 1,
                                            game.zoom - 2, game.zoom - 2])
                            
            
        # font = pygame.font.SysFont('Microsoft JhengHei', 25, True, False)
        # attack text
        font = pygame.font.Font("./src/static/NotoSansTC-Bold.ttf", 30)
        text_combo = font.render(str(game.attack_combo - game.combo3), True, RED)
        if game.attacking:
            text_attack = font.render("攻擊 ! ", True, BLACK)
        else:
            text_attack = font.render("再消除      行即可攻擊！", True, BLACK)

        # voice text
        font_voice = pygame.font.Font("./src/static/NotoSansTC-Bold.ttf", 65)
        text_voice = font_voice.render(str(txt), True, BLACK)

        # game over text
        font1 = pygame.font.SysFont('Calibri', 65, True, False)
        text_game_over = font1.render("Game Over", True, (255, 125, 0))
        text_game_over1 = font1.render("Press ESC", True, (255, 215, 0))

        screen.blit(text_combo, [180, 20])
        screen.blit(text_attack, [80, 20])
        screen.blit(text_voice, [720, 750])
        if game.state == "gameover":
            pygame.draw.rect(screen, (0, 0, 0), [0, 0, size[0], size[1]])
            screen.blit(text_game_over, [650, 200])
            screen.blit(text_game_over1, [650, 265])

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
