import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

DEBUG_MODE = True
MIN_THUMB_HEIGHT_DIFFERENCE = 0.1

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = hands.process(image)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    gesture = "Nenhum gesto reconhecido"
    debug_text = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            
            diff_index = thumb_tip.y - index_tip.y
            diff_middle = thumb_tip.y - middle_tip.y
            diff_ring = thumb_tip.y - ring_tip.y
            diff_pinky = thumb_tip.y - pinky_tip.y
            
            if DEBUG_MODE:
                debug_text.append(f"Polegar Y: {thumb_tip.y:.2f}")
                debug_text.append(f"Indicador Y: {index_tip.y:.2f} (Diff: {diff_index:.2f})")
                debug_text.append(f"Medio Y: {middle_tip.y:.2f} (Diff: {diff_middle:.2f})")
                debug_text.append(f"Anelar Y: {ring_tip.y:.2f} (Diff: {diff_ring:.2f})")
                debug_text.append(f"Mindinho Y: {pinky_tip.y:.2f} (Diff: {diff_pinky:.2f})")
            
            is_thumb_up = (
                diff_index < -MIN_THUMB_HEIGHT_DIFFERENCE and
                diff_middle < -MIN_THUMB_HEIGHT_DIFFERENCE and
                diff_ring < -MIN_THUMB_HEIGHT_DIFFERENCE and
                diff_pinky < -MIN_THUMB_HEIGHT_DIFFERENCE
            )
            
            if is_thumb_up:
                index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                
                fingers_closed = (
                    abs(index_tip.y - index_mcp.y) < 0.1 and
                    abs(middle_tip.y - middle_mcp.y) < 0.1
                )
                
                if fingers_closed:
                    gesture = "Joinha detectado!"
                else:
                    gesture = "Polegar para cima, mas dedos abertos"
            else:
                gesture = "Nao e joinha"
    
    cv2.putText(image, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if DEBUG_MODE:
        for i, text in enumerate(debug_text):
            cv2.putText(image, text, (10, 60 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.imshow('Reconhecimento Gestual', image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()