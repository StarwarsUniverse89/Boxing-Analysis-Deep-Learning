import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.models import load_model

def detect_boxer(frame, model, threshold=0.5):
    height, width, channels = frame.shape
    blob = cv2.resize(frame, (224, 224))
    blob = np.expand_dims(blob, axis=0)
    blob = blob / 255.0
    
    prediction = model.predict(blob)
    
    if prediction[0][0] > threshold:
        x_center = width // 2
        y_center = height // 2
        return True, x_center, y_center
    return False, 0, 0

def track_positions(base_folder, model, threshold=0.5):
    positions = []

    for subdir, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".jpg"):
                frame_path = os.path.join(subdir, file)
                frame = cv2.imread(frame_path)
                detected, x_center, y_center = detect_boxer(frame, model, threshold)

                if detected:
                    positions.append((x_center, y_center))

    return positions

def convert_2d_to_3d(positions, ring_length=10, ring_width=10, ring_height=2):
    positions_3d = []
    for x, y in positions:
        x_3d = x / 100 * ring_length
        y_3d = y / 100 * ring_width
        z_3d = ring_height / 2
        positions_3d.append((x_3d, y_3d, z_3d))
    return positions_3d

def create_3d_heatmap(positions_3d):
    x_positions = [pos[0] for pos in positions_3d]
    y_positions = [pos[1] for pos in positions_3d]
    z_positions = [pos[2] for pos in positions_3d]

    heatmap, xedges, yedges = np.histogram2d(x_positions, y_positions, bins=(100, 100))

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.5, yedges[:-1] + 0.5, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    dx = dy = 1 * np.ones_like(zpos)
    dz = heatmap.ravel()

    cmap = plt.get_cmap('hot')
    max_height = np.max(dz)
    min_height = np.min(dz)
    rgba = [cmap((k - min_height) / max_height) for k in dz]

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Density')
    ax.set_title("3D Heatmap of Famous Boxer's Movements")
    plt.show()

if __name__ == "__main__":
    model_path = "/content/drive/My Drive/BoxingModel/model_retrained.h5"
    model = load_model(model_path)

    boxing_ml_folder = "/content/drive/My Drive/BoxingML"
    positions = track_positions(boxing_ml_folder, model, threshold=0.2)
    positions_3d = convert_2d_to_3d(positions)
    create_3d_heatmap(positions_3d)
