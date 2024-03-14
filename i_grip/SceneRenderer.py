
from i_grip.Hands_refactored import GraspingHand
from i_grip.Objects import RigidObject

class SceneRenderer:
    def __init__(self) -> None:
        self.hands_data = {}
        self.objects_data = {}
        self.fps_data = {}
    
    def set_hands_data(self, hands_data):
        self.hands_data = hands_data
    
    def set_objects_data(self, objects_data):
        self.objects_data = objects_data
    
    def set_fps_data(self, fps_data):
        self.fps_data = fps_data
    
    def render_hands(self, img):
        for hand_data in self.hands_data.values():
            if hand_data is not None:
                GraspingHand.render_hand(img, **hand_data)
    
    
    def render_objects(self, img):
        for obj_data in self.objects_data.values():
            if obj_data is not None:
                RigidObject.render_object(img, **obj_data)
    
    def render(self, img):
        self.render_hands(img)
        self.render_objects(img)    
        