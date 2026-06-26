#!/usr/bin/env python3
"""Focus Tracker - Enterprise Cognitive State, Drowsiness & Live Data Graph Monitoring System"""
import cv2, numpy as np, mediapipe as mp, time, os, sys, csv
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

# Configuration & HUD Colors (BGR)
GX_TH, GY_TH, Y_TH, P_TH, EAR_TH, MAR_TH, YAWN_TH, LKD_TH = 0.15, 0.15, 0.15, 0.15, 0.18, 0.50, 1.5, 10.0
C_GRN, C_ORG, C_RED, C_CYN, C_WHT, C_GRY, C_DRK = (120,200,80), (34,126,230), (60,76,231), (230,230,0), (255,255,255), (150,150,150), (15,15,15)
L_EYE, R_EYE, L_IRS, R_IRS = [33,160,158,133,153,144], [362, 385, 387, 263, 373, 380], 468, 473

def dist(p1, p2, w, h):
    return np.hypot((p1.x - p2.x)*w, (p1.y - p2.y)*h)

def eye_metrics(lms, idxs, iris, w, h):
    pts = np.array([(lms[i].x*w, lms[i].y*h) for i in idxs])
    cx, cy = np.mean(pts, axis=0)
    ew, eh = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
    return (lms[iris].x*w - cx)/ew if ew > 0 else 0, (lms[iris].y*h - cy)/eh if eh > 0 else 0

def get_ear(lms, idxs, w, h):
    d1 = dist(lms[idxs[1]], lms[idxs[4]], w, h)
    d2 = dist(lms[idxs[2]], lms[idxs[5]], w, h)
    dh = dist(lms[idxs[0]], lms[idxs[3]], w, h)
    return (d1 + d2) / (2.0 * dh) if dh > 0 else 0

def get_mar(lms, w, h):
    d1, d2, d3 = dist(lms[82], lms[87], w, h), dist(lms[13], lms[14], w, h), dist(lms[312], lms[317], w, h)
    dh = dist(lms[78], lms[308], w, h)
    return (d1 + d2 + d3) / (2.0 * dh) if dh > 0 else 0

def head_pose(lms, w, h):
    dl, dr = dist(lms[1], lms[234], w, h), dist(lms[1], lms[454], w, h)
    df, dc = dist(lms[1], lms[10], w, h), dist(lms[1], lms[152], w, h)
    return (dl - dr)/(dl + dr) if (dl+dr) > 0 else 0, (df - dc)/(df+dc) if (df+dc) > 0 else 0

def draw_hud(img, p1, p2, c, a):
    ov = img.copy()
    cv2.rectangle(ov, p1, p2, c, -1)
    cv2.addWeighted(ov, a, img, 1-a, 0, img)

def draw_bracket(img, pt1, pt2, c, t=2, d=15):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1, y1), (x2, y2), c, 1, cv2.LINE_AA)
    for x, y, dx, dy in [(x1,y1,d,d), (x2,y1,-d,d), (x1,y2,d,-d), (x2,y2,-d,-d)]:
        cv2.line(img, (x, y), (x + dx, y), c, t, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + dy), c, t, cv2.LINE_AA)

def get_camera():
    for idx in range(3):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened() and cap.read()[0]: return cap
    return cv2.VideoCapture(0)

def main():
    # Resolve resource path for PyInstaller or local workspace
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_path, 'face_landmarker.task')
    if not os.path.exists(model_path):
        model_path = 'face_landmarker.task'
        
    if not os.path.exists(model_path):
        sys.exit(print(f"Error: {model_path} missing"))
        
    det = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE, num_faces=4
    ))
    cap = get_camera()
    ox, oy, oyaw, opit = 0.0, 0.0, 0.0, 0.0
    yawn_t, closed_t, lkd_t, cal_end, sm_score = None, None, None, 0.0, 95.0
    wname = "Focus & Drowsiness Tracker HUD"
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    
    ret, frame = cap.read()
    if ret: cv2.resizeWindow(wname, frame.shape[1], frame.shape[0])
    
    # Focus Index data history array for scrolling graph
    score_history = []
    
    # Excel logging directory configuration
    log_dir = "C:\\Sharook project"
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning: Directory creation failed: {e}")
        
    log_file = os.path.join(log_dir, "focus_telemetry_log.csv")
    last_log_time = time.time()
    
    # Initialize in-memory telemetry log buffer
    log_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        results = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        
        fc = len(results.face_landmarks) if results.face_landmarks else 0
        face_ok, gx, gy, yaw, pitch, mar, l_ear, r_ear, yawn_dur, closed_dur, l_op, r_op = False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.0, 0.0, False, False
        state, score, c_state, feeling = "Focused", 95.0, C_GRN, "Energetic / Focused"
        
        if fc > 0:
            face_ok = True
            for i in range(fc):
                lms = results.face_landmarks[i]
                xs, ys = [lm.x*w for lm in lms], [lm.y*h for lm in lms]
                x1, x2 = max(0, int(min(xs) - (max(xs)-min(xs))*0.05)), min(w, int(max(xs) + (max(xs)-min(xs))*0.05))
                y1, y2 = max(0, int(min(ys) - (max(ys)-min(ys))*0.05)), min(h, int(max(ys) + (max(ys)-min(ys))*0.05))
                if i > 0:
                    draw_bracket(frame, (x1, y1), (x2, y2), C_RED)
                    lbl = f"INTRUDER {i}"
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    bx1, bx2 = int((x1+x2)/2 - tw/2 - 10), int((x1+x2)/2 + tw/2 + 10)
                    by1, by2 = y1 - th - 15, y1 - 5
                    if by1 < 0: by1, by2 = y2 + 5, y2 + th + 15
                    draw_hud(frame, (bx1, by1), (bx2, by2), C_DRK, 0.6)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), C_RED, 1, cv2.LINE_AA)
                    cv2.putText(frame, lbl, (bx1 + 10, by2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHT, 1, cv2.LINE_AA)
            
            p_lms = results.face_landmarks[0]
            raw_y, raw_p = head_pose(p_lms, w, h)
            yaw, pitch = raw_y - oyaw, raw_p - opit
            l_ear = get_ear(p_lms, L_EYE, w, h)
            r_ear = get_ear(p_lms, R_EYE, w, h)
            l_op, r_op = l_ear > EAR_TH, r_ear > EAR_TH
            
            gxl, gyl = eye_metrics(p_lms, L_EYE, L_IRS, w, h)
            gxr, gyr = eye_metrics(p_lms, R_EYE, R_IRS, w, h)
            if l_op and r_op: gx, gy = (gxl + gxr)/2.0 - ox, (gyl + gyr)/2.0 - oy
            elif l_op: gx, gy = gxl - ox, gyl - oy
            elif r_op: gx, gy = gxr - ox, gyr - oy
            
            mar = get_mar(p_lms, w, h)
            
            g_dev = np.hypot(gx/GX_TH, gy/GY_TH)
            h_dev = np.hypot(yaw/Y_TH, pitch/P_TH)
            tot_dev = min(1.0, (g_dev + h_dev) / 2.0)
            focused_score = 91.0 + 8.0 * (1.0 - tot_dev)
            
            if fc > 1:
                state, score, c_state = "Distracted", 0.0, C_RED
                feeling = "Biometric Security Lockdown"
                yawn_t = closed_t = None
            elif abs(yaw) > Y_TH or abs(pitch) > P_TH or ((l_op or r_op) and (abs(gx) > GX_TH or abs(gy) > GY_TH)):
                state, score, c_state = "Distracted", 0.0, C_RED
                feeling = "Disengaged / Looked Away"
                yawn_t = closed_t = None
            elif not l_op and not r_op:
                if closed_t is None: closed_t = time.time()
                closed_dur = time.time() - closed_t
                if closed_dur > 1.5:
                    state, score, c_state = "Drowsy", 30.0, C_ORG
                    feeling = "Unconscious / Micro-sleeping!"
                else:
                    feeling = "Blinking / Resting"
            else:
                closed_t = None
                if mar > MAR_TH:
                    if yawn_t is None: yawn_t = time.time()
                    yawn_dur = time.time() - yawn_t
                    if yawn_dur > YAWN_TH:
                        state, score, c_state = "Drowsy", 30.0, C_ORG
                        feeling = "Extremely Fatigued / Yawning"
                    else:
                        state, score, c_state = "Focused", focused_score, C_GRN
                        feeling = "Sluggish / Starting to Yawn"
                else:
                    yawn_t = None
                    state, score, c_state = "Focused", focused_score, C_GRN
                    if tot_dev < 0.3:
                        feeling = "Deeply Engaged & Concentrated"
                    elif tot_dev < 0.7:
                        feeling = "Active / Normal Focus"
                    else:
                        feeling = "Relaxed / Loose Focus"
            
            for i in L_EYE + R_EYE: cv2.circle(frame, (int(p_lms[i].x*w), int(p_lms[i].y*h)), 1, (230,230,230), -1, cv2.LINE_AA)
            if l_op: cv2.circle(frame, (int(p_lms[L_IRS].x*w), int(p_lms[L_IRS].y*h)), 2, C_CYN, -1, cv2.LINE_AA)
            if r_op: cv2.circle(frame, (int(p_lms[R_IRS].x*w), int(p_lms[R_IRS].y*h)), 2, C_CYN, -1, cv2.LINE_AA)
            
            xs, ys = [lm.x*w for lm in p_lms], [lm.y*h for lm in p_lms]
            x1, x2 = max(0, int(min(xs) - (max(xs)-min(xs))*0.05)), min(w, int(max(xs) + (max(xs)-min(xs))*0.05))
            y1, y2 = max(0, int(min(ys) - (max(ys)-min(ys))*0.05)), min(h, int(max(ys) + (max(ys)-min(ys))*0.05))
            draw_bracket(frame, (x1, y1), (x2, y2), c_state)
            
            lbl = "LOCKDOWN ACTIVE" if fc>1 else ("WARNING: Drowsy" if state=="Drowsy" else f"Status: {state}")
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            bx1, bx2 = int((x1+x2)/2 - tw/2 - 10), int((x1+x2)/2 + tw/2 + 10)
            by1, by2 = y1 - th - 15, y1 - 5
            if by1 < 0: by1, by2 = y2 + 5, y2 + th + 15
            draw_hud(frame, (bx1, by1), (bx2, by2), C_DRK, 0.6)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), c_state, 1, cv2.LINE_AA)
            cv2.putText(frame, lbl, (bx1 + 10, by2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHT, 1, cv2.LINE_AA)
            
            feel_lbl = f"Feeling: {feeling}"
            (ftw, fth), _ = cv2.getTextSize(feel_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            fbx1, fbx2 = int((x1+x2)/2 - ftw/2 - 8), int((x1+x2)/2 + ftw/2 + 8)
            fby1, fby2 = by2 + 3, by2 + fth + 11
            if fby2 > h: fby1, fby2 = by1 - fth - 11, by1 - 3
            draw_hud(frame, (fbx1, fby1), (fbx2, fby2), C_DRK, 0.5)
            cv2.rectangle(frame, (fbx1, fby1), (fbx2, fby2), (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(frame, feel_lbl, (fbx1 + 8, fby2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_WHT, 1, cv2.LINE_AA)
            
        else:
            state, score, c_state, feeling = "Distracted", 0.0, C_RED, "No Face Detected"
            lbl = "FACE DETECTOR OFFLINE"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            tx, ty = int(w/2 - tw/2), int(h/2 + th/2)
            draw_hud(frame, (tx - 15, ty - th - 15), (tx + tw + 15, ty + 15), C_DRK, 0.6)
            cv2.rectangle(frame, (tx - 15, ty - th - 15), (tx + tw + 15, ty + 15), C_RED, 1, cv2.LINE_AA)
            cv2.putText(frame, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_RED, 2, cv2.LINE_AA)
            
        sm_score += (score - sm_score) * 0.15
        ds = int(np.clip(sm_score, 0, 100))
        
        # Add to sliding Focus Index history buffer
        score_history.append(float(sm_score))
        if len(score_history) > 100:
            score_history.pop(0)
            
        # Minimized Telemetry Card Panel (Bottom-Right)
        panel_w, panel_h = 210, 132
        x1, y1 = w - panel_w - 15, h - panel_h - 15
        x2, y2 = w - 15, h - 15
        draw_hud(frame, (x1, y1), (x2, y2), C_DRK, 0.7)
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_GRY, 1, cv2.LINE_AA)
        
        cv2.putText(frame, "TELEMETRY HUD", (x1 + 12, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_WHT, 1, cv2.LINE_AA)
        cv2.line(frame, (x1 + 12, y1 + 24), (x2 - 12, y1 + 24), (70, 70, 70), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Focus Index: {ds}%", (x1 + 12, y1 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_WHT, 1, cv2.LINE_AA)
        
        bar_x1, bar_y1 = x1 + 12, y1 + 46
        bar_x2, bar_y2 = x2 - 12, y1 + 54
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (40, 40, 40), -1)
        bw = int((bar_x2 - bar_x1) * (ds / 100.0))
        bc = C_GRN if ds >= 70 else (C_ORG if ds >= 30 else C_RED)
        if bw > 0: cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bw, bar_y2), bc, -1)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), C_GRY, 1, cv2.LINE_AA)
        
        feeling_short = feeling[:23] + ".." if len(feeling) > 25 else feeling
        cv2.putText(frame, f"State: {state}", (x1 + 12, y1 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c_state, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Feel : {feeling_short}", (x1 + 12, y1 + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.33, C_WHT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Gaze : X:{gx:+.1f} Y:{gy:+.1f}", (x1 + 12, y1 + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_GRY, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Pose : Y:{yaw:+.1f} P:{pitch:+.1f}", (x1 + 12, y1 + 118), cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_GRY, 1, cv2.LINE_AA)
        
        # Tiny Radar crosshair Map
        cx, cy = x2 - 32, y1 + 95
        cv2.circle(frame, (cx, cy), 14, (40, 40, 40), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 14, C_GRY, 1, cv2.LINE_AA)
        cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (60, 60, 60), 1)
        cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (60, 60, 60), 1)
        if l_op or r_op: cv2.circle(frame, (int(cx + np.clip(gx/GX_TH, -1.0, 1.0)*11), int(cy + np.clip(gy/GY_TH, -1.0, 1.0)*11)), 2, C_CYN, -1, cv2.LINE_AA)
        
        # Live Focus Index Data Graph HUD Panel (Top-Left)
        p_w, p_h = 210, 95
        px1, py1 = 15, 15
        px2, py2 = 15 + p_w, 15 + p_h
        draw_hud(frame, (px1, py1), (px2, py2), C_DRK, 0.7)
        cv2.rectangle(frame, (px1, py1), (px2, py2), C_GRY, 1, cv2.LINE_AA)
        
        cv2.putText(frame, "FOCUS MONITOR", (px1 + 12, py1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_WHT, 1, cv2.LINE_AA)
        cv2.line(frame, (px1 + 12, py1 + 24), (px2 - 12, py1 + 24), (70, 70, 70), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Live Index: {ds}%", (px1 + 12, py1 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_WHT, 1, cv2.LINE_AA)
        
        # Render scrolling Focus Data Waveform
        gx_x1, gx_y1 = px1 + 12, py1 + 45
        gx_x2, gx_y2 = px2 - 12, py2 - 12
        cv2.rectangle(frame, (gx_x1, gx_y1), (gx_x2, gx_y2), (25, 25, 25), -1)
        cv2.rectangle(frame, (gx_x1, gx_y1), (gx_x2, gx_y2), (50, 50, 50), 1, cv2.LINE_AA)
        
        hist_len = len(score_history)
        if hist_len > 2:
            seg_len = min(80, hist_len)
            score_seg = score_history[-seg_len:]
            
            pts = []
            for idx_pt, val_pt in enumerate(score_seg):
                pt_x = gx_x2 - 3 - int((seg_len - 1 - idx_pt) * (180.0 / seg_len))
                pt_y = gx_y2 - 3 - int(val_pt / 100.0 * 32)
                pts.append((pt_x, pt_y))
                
            for idx_pt in range(len(pts) - 1):
                cv2.line(frame, pts[idx_pt], pts[idx_pt+1], c_state, 1, cv2.LINE_AA)
                
        # Save results to memory buffer once per second (continuous logging including face-offline idle)
        curr_time = time.time()
        if curr_time - last_log_time >= 1.0:
            last_log_time = curr_time
            # If no face is detected, log EAR values as 0.0000 for clarity
            log_l_ear = 0.0 if not face_ok else l_ear
            log_r_ear = 0.0 if not face_ok else r_ear
            log_buffer.append([
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "TEMP_SUBJECT",
                f"{ds}%", state, feeling,
                f"{gx:+.4f}", f"{gy:+.4f}", f"{yaw:+.4f}", f"{pitch:+.4f}",
                f"{log_l_ear:.4f}", f"{log_r_ear:.4f}", fc
            ])
                
        cv2.putText(frame, "Press 'C' to calibrate center gaze | 'Q' to Exit", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_WHT, 1, cv2.LINE_AA)
        
        if fc > 1:
            if lkd_t is None: lkd_t = time.time()
            rem = max(0.0, LKD_TH - (time.time() - lkd_t))
            bx1, by1 = int(w/2 - 210), int(h/2 - 80)
            bx2, by2 = int(w/2 + 210), int(h/2 + 80)
            f_color = C_RED if int(time.time() * 2) % 2 == 0 else (20,20,20)
            draw_hud(frame, (bx1, by1), (bx2, by2), f_color, 0.8)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), C_RED, 2, cv2.LINE_AA)
            cv2.putText(frame, "SECURITY LOCKDOWN ALERT!", (int(w/2 - 130), by1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHT, 2, cv2.LINE_AA)
            cv2.putText(frame, f"MULTIPLE FACES DETECTED: {fc} IN FRAME", (int(w/2 - 145), by1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRY, 1, cv2.LINE_AA)
            cv2.putText(frame, f"AUTO-TERMINATION IN: {rem:.1f}s", (int(w/2 - 135), by1 + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_WHT, 2, cv2.LINE_AA)
            cv2.putText(frame, "Please clear the camera frame immediately!", (int(w/2 - 140), by2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRY, 1, cv2.LINE_AA)
            if rem <= 0.0:
                print("[CRITICAL] Security shutdown triggered.")
                break
        else:
            lkd_t = None
            
        if time.time() < cal_end:
            banner = "GAZE CALIBRATION SUCCESSFUL!"
            (btw, bth), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            draw_hud(frame, (int(w/2 - btw/2 - 10), 20), (int(w/2 + btw/2 + 10), 50), C_GRN, 0.8)
            cv2.putText(frame, banner, (int(w/2 - btw/2), 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_WHT, 2, cv2.LINE_AA)
            cv2.circle(frame, (int(w/2), int(h/2)), 5, C_GRN, 1, cv2.LINE_AA)
            cv2.circle(frame, (int(w/2), int(h/2)), 2, C_GRN, -1, cv2.LINE_AA)
            
        cv2.imshow(wname, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('c'), ord('C')) and face_ok:
            ox, oy = (gxl + gxr)/2.0, (gyl + gyr)/2.0
            oyaw, opit = raw_y, raw_p
            cal_end = time.time() + 1.5
            print(f"[CALIBRATION] Baseline updated: GazeOffset=({ox:.4f},{oy:.4f}), HeadOffset=({oyaw:.4f},{opit:.4f})")
        elif key in (ord('q'), ord('Q'), 27):
            break
            
        # Check if user closed the window via the window manager 'X' button
        if cv2.getWindowProperty(wname, cv2.WND_PROP_VISIBLE) < 1:
            break
            
    # Release camera and GUI resources immediately so the window closes
    cap.release()
    cv2.destroyAllWindows()
    det.close()
    
    if not log_buffer:
        print("[INFO] No telemetry data was recorded during this session.")
        return
        
    # Count existing CSV files to auto-number defaults (e.g. Person 1, Person 2...)
    session_num = 1
    try:
        if os.path.exists(log_dir):
            csv_files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
            session_num += len(csv_files)
    except Exception:
        pass
            
    default_name = f"Person {session_num}"
    participant_name = default_name
    
    # Prompt for participant name - works in IDLE, VS Code, Colab, Command Prompt
    print("\n" + "="*50)
    print("SESSION TRACKING COMPLETED")
    print(f"Default Auto-Label: {default_name}")
    print("Please enter your name to save the recorded session data.")
    print("Press ENTER to save as the default auto-label.")
    print("="*50)
    
    try:
        # Standard input works fine here since GUI window has already been closed
        name_input = input("Enter Subject Name: ").strip()
        if name_input:
            participant_name = name_input
    except Exception as e:
        print(f"Using default label due to input prompt error: {e}")
        
    # Replace placeholder with the actual name in the buffer
    for row in log_buffer:
        if row[1] == "TEMP_SUBJECT":
            row[1] = participant_name
            
    # Format a filename-safe string from participant name
    safe_name = "".join(c for c in participant_name if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
    if not safe_name:
        safe_name = "Subject"
        
    # Generate a unique filename using participant name and timestamp
    timestamp_str = time.strftime('%Y%m%d_%H%M%S')
    log_file_unique = os.path.join(log_dir, f"{safe_name}_focus_log_{timestamp_str}.csv")
    
    # Prepare final rows (always write headers for the new session file)
    final_rows = []
    final_rows.append([
        "Timestamp", "Participant", "Focus_Score", "State", "Feeling", 
        "Gaze_X", "Gaze_Y", "Head_Yaw", "Head_Pitch", 
        "Left_EAR", "Right_EAR", "Face_Count"
    ])
    final_rows.extend(log_buffer)
    
    # Attempt to write to the unique session file with Excel lock protection
    saved = False
    while not saved:
        try:
            with open(log_file_unique, 'w', newline='') as f_log:
                writer = csv.writer(f_log)
                writer.writerows(final_rows)
            saved = True
            print(f"\n[SUCCESS] Telemetry data successfully saved to: {log_file_unique}")
            print(f"Recorded subject: {participant_name}\n")
        except Exception as e:
            print(f"\n[ERROR] Failed to save data: {e}")
            print(f"The file {log_file_unique} might be open or directory is write-protected.")
            print("Please close any conflicting application and press ENTER to try saving again, or type 'q' to abort.")
            try:
                ans = input("Press ENTER to retry (or 'q' to abort): ").strip().lower()
                if ans == 'q':
                    print("[ABORTED] Session data was not saved.")
                    break
            except Exception:
                # If input prompt fails during error recovery, break to prevent infinite hang
                print("[ABORTED] Session data was not saved due to input failure.")
                break

if __name__ == "__main__":
    main()
