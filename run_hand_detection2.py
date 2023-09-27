import argparse
from i_grip import HandDetectors2 as hd
from i_grip import Scene2 as sc
import torch
import gc
import os
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   print(torch.cuda.memory_snapshot())
   torch.cuda.empty_cache()

class HandDetector:
    def __init__(self) -> None:
        
        self.hand_detector = hd.HybridOAKMediapipeDetector()
        device_data = self.hand_detector.get_device_data()
        self.scene = sc.LiveScene(device_data, name = 'Full tracking')
        self.img = None


    def run(self):
        print(self.__dict__)
        self.hand_detector.start()
        print('start')
        success, img = self.hand_detector.next_frame()
        i=0
        while self.hand_detector.isOn():
            # print('new_image_event.is_set()',new_image_event.is_set())
            success, img = self.hand_detector.next_frame()
            if not success:
                self.img = None
                continue     
            else:
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
                img.flags.writeable = False
                hands = self.hand_detector.get_hands(img)
                if hands is not None and len(hands)>0:
                    self.scene.update_hands(hands)
                img.flags.writeable = True
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            self.scene.render(img)
            cv2.imshow(f'view',img)
            k = cv2.waitKey(1)
            if k==27:
                print('end')
                break
            # if i>10:
            #     break
            # i+=1
        self.scene.stop()
        cv2.destroyAllWindows()
        exit()


        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-hd', '--hand_detection', choices=['mediapipe', 'depthai', 'hybridOAKMediapipe'],
                        default = 'hybridOAKMediapipe', help="Hand pose reconstruction solution")
    parser.add_argument('-od', '--object_detection', choices=['cosypose, megapose'],
                        default = 'cosypose', help="Object pose reconstruction detection")
    args = vars(parser.parse_args())

    # if args.hand_detection == 'mediapipe':
    #     import mediapipe as mp
    # else:
    #     import depthai as dai
    
    # if args.object_detection == 'cosypose':
    #     import cosypose
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    report_gpu()
    i_grip = HandDetector()
    i_grip.run()