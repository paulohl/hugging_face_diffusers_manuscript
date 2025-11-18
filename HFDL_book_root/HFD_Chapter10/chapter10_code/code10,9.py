# enhanced training with frame skipping: 

def preprocess_state(state):
    gray = np.mean(state, axis=2)
    resized = cv2.resize(gray, (84, 84))
    return resized / 255.0
