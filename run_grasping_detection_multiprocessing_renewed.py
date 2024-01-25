import multiprocessing
import cv2
import time


from i_grip import HandDetectors2 as hd
from i_grip import Object2DDetectors as o2d
from i_grip import ObjectPoseEstimators as ope
from i_grip import Scene2 as sc
from i_grip import Plotters3 as pl
from i_grip.utils import kill_gpu_processes

class GraspingDetector:
    def __init__(self) -> None:
        dataset = "ycbv"
        self.hand_detector = hd.HybridOAKMediapipeDetector()
        cam_data = self.hand_detector.get_device_data()
        plotter = pl.NBPlot()
        self.object_detector = o2d.get_object_detector(dataset, cam_data)
        self.object_pose_estimator = ope.get_pose_estimator(dataset, cam_data, use_tracking=True, fuse_detections=False)
        self.scene = sc.LiveScene(cam_data, name='Full tracking', plotter=plotter)
        self.object_detections = None
        self.is_hands = False
        self.img_for_objects = None
        
    def get_img(self, output_queue):
        while True:
            my_img = output_queue.get()
            self.img_for_objects = my_img
        
    def detect_hands(self, input_queue, output_queue):
        while True:
            my_img = input_queue.get()
            # Appliquer la fonction detect_hands sur my_img
            detected_hands = detect_hands_function(my_img)
            output_queue.put(detected_hands)

    def detect_objects(self, input_queue, output_queue):
        while True:
            my_img = input_queue.get()
            # Appliquer la fonction detect_objects sur my_img
            detected_objects = detect_objects_function(my_img)
            output_queue.put(detected_objects)

    def estimate_objects(self, input_queue, output_queue):
        while True:
            detected_objects = input_queue.get()
            my_img = input_queue.get()
            # Appliquer la fonction estimate_objects sur my_img et detected_objects
            estimated_objects = estimate_objects_function(my_img, detected_objects)
            output_queue.put(estimated_objects)

    def analysis(self, input_queue_hands, input_queue_objects, input_queue_estimated_objects):
        while True:
            my_img = input_queue_hands.get()
            detected_hands = input_queue_objects.get()
            estimated_objects = input_queue_estimated_objects.get()
            # Déclencher la fonction update_scene avec my_img, detected_hands et estimated_objects
            update_scene_function(my_img, detected_hands, estimated_objects)

    def run(self):
        # Initialiser les queues pour la communication entre les processus
        input_queue_hands = multiprocessing.Queue(maxsize=1)
        input_queue_objects = multiprocessing.Queue(maxsize=1)
        input_queue_estimated_objects = multiprocessing.Queue(maxsize=1)

        # Créer et démarrer les processus
        process_hands = multiprocessing.Process(target=detect_hands, args=(input_queue_hands, input_queue_objects))
        process_objects = multiprocessing.Process(target=detect_objects, args=(input_queue_objects, input_queue_estimated_objects))
        process_estimation = multiprocessing.Process(target=estimate_objects, args=(input_queue_objects, input_queue_estimated_objects))
        process_analysis = multiprocessing.Process(target=analysis, args=(input_queue_hands, input_queue_objects, input_queue_estimated_objects))

        process_hands.start()
        process_objects.start()
        process_estimation.start()
        process_analysis.start()

        # Boucle principale
        for i in range(5):  # Exemple : boucler sur 5 images
            my_img = cv2.imread(f'image_{i}.jpg')  # Charger une image (à remplacer par votre code de lecture d'image)
            
            # Mettre l'image dans les queues pour les processus correspondants
            input_queue_hands.put(my_img)
            input_queue_objects.put(my_img)

            # Attendre les résultats des processus
            detected_hands = input_queue_objects.get()
            input_queue_hands.put(detected_hands)  # Envoyer le résultat à analysis

            estimated_objects = input_queue_estimated_objects.get()
            input_queue_hands.put(estimated_objects)  # Envoyer le résultat à analysis

            time.sleep(2)  # Simuler un certain délai entre les images

        # Terminer les processus
        process_hands.terminate()
        process_objects.terminate()
        process_estimation.terminate()
        process_analysis.terminate()

# Simuler les fonctions pour la détection et l'estimation (à remplacer par vos fonctions réelles)
def detect_hands_function(img):
    time.sleep(1)  # Simulation d'une opération longue
    return f"Hands detected in {img}"

def detect_objects_function(img):
    time.sleep(1)  # Simulation d'une opération longue
    return f"Objects detected in {img}"

def estimate_objects_function(img, detected_objects):
    time.sleep(1)  # Simulation d'une opération longue
    return f"Objects estimated in {img} with {detected_objects}"

def update_scene_function(img, detected_hands, estimated_objects):
    print(f"Updating scene with {img}, {detected_hands}, {estimated_objects}")
